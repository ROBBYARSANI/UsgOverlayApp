package com.example.usgandroidapp;

import android.app.Activity;
import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.PendingIntent;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;

import androidx.core.app.NotificationCompat;

import java.io.File;
import java.io.FileOutputStream;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/**
 * VeinDetectionService.java — YOLOv8-Seg Edition
 * ================================================
 * Service ini menangani:
 *   1. Screen capture via MediaProjection
 *   2. Frame pipeline: Camera thread → AtomicReference → Inference thread
 *   3. Adaptive EMA smoothing agar overlay "lengket" saat probe bergerak
 *   4. Anti-blink hold frames
 *   5. Render hasil ke OverlayView via MainHandler
 *
 * ARSITEKTUR THREADING:
 *
 *   Camera Thread (HandlerThread)          Inference Thread (SingleThreadExecutor)
 *   ─────────────────────────────          ───────────────────────────────────────
 *   acquireLatestImage()                   ← mLatestFrame.getAndSet(null)
 *   decode → Bitmap                        ← YoloDetector.detect()
 *   mLatestFrame.getAndSet(newFrame)       ← Adaptive EMA smooth
 *   recycle frame lama yang terbuang       ← postAtFrontOfQueue → OverlayView
 *   trigger jika AI idle                   ← cek mLatestFrame lagi (self-schedule)
 *
 * AI SELALU proses frame TERBARU — frame lama dibuang otomatis tanpa antrian.
 */
public class VeinDetectionService extends Service {

    private static final String TAG = "VeinDetectionService";

    public static final String ACTION_START = "com.example.usgandroidapp.ACTION_START";
    public static final String ACTION_STOP  = "com.example.usgandroidapp.ACTION_STOP";
    public static final String ACTION_CAPTURE_MODEL_INPUT =
            "com.example.usgandroidapp.CAPTURE_MODEL_INPUT";
    public static final String EXTRA_CAPTURE_MODE = "capture_mode";

    public static final String EXTRA_RESULT_CODE     = "extra_result_code";
    public static final String EXTRA_PROJECTION_DATA = "extra_projection_data";

    private static final String NOTIFICATION_CHANNEL_ID = "vein_detection_channel";
    private static final int    NOTIFICATION_ID          = 101;

    private static volatile boolean              sIsRunning = false;
    private static volatile VeinDetectionService sInstance  = null;

    // Selected after clean replay A/B: old FP32 detector + minimal anti-blink only.
    public static volatile TemporalMode sTemporalMode = TemporalMode.MINIMAL_ANTIBLINK;

    private MediaProjectionManager mProjectionManager;
    private MediaProjection        mMediaProjection;
    private FrameSource            mFrameSource;
    private WindowManager          mWindowManager;
    private OverlayView            mOverlayView;
    private YoloDetector           mYoloDetector;
    private VeinOpticalFlowTracker mTracker;
    private VeinKalmanTracker      mKalmanTracker;
    private AntiBlinkStateMachine  mAntiBlinkStateMachine;
    private android.content.BroadcastReceiver mModeReceiver;

    private HandlerThread mProcessingThread;
    private Handler       mProcessingHandler;
    private Handler       mMainHandler;

    private final ExecutorService mInferenceExecutor = Executors.newSingleThreadExecutor();

    private int     mScreenWidth;
    private int     mScreenHeight;
    private int     mScreenDensity;
    private boolean mOverlayVisible = true;
    private boolean mCleanupDone    = false;

    // ─────────────────────────────────────────────────────────────
    // LATEST-FRAME DROPPING PIPELINE
    // ─────────────────────────────────────────────────────────────
    private static class FramePacket {
        final Bitmap bitmap;
        final long captureTimestampNs;
        FramePacket(Bitmap b, long ts) {
            this.bitmap = b;
            this.captureTimestampNs = ts;
        }
    }
    /** Slot frame tunggal: camera thread menulis, inference thread membaca & mengosongkan. */
    private final AtomicReference<FramePacket> mLatestFrame = new AtomicReference<>(null);
    /** True = inference sedang berjalan, jangan spawn inference baru secara paralel. */
    private final AtomicBoolean                mIsInferring = new AtomicBoolean(false);
    /** One-shot audit request. The saved PNG is the exact Bitmap passed to detect(). */
    private final AtomicReference<String>      mPendingInputAudit = new AtomicReference<>(null);

    // ─────────────────────────────────────────────────────────────
    // ADAPTIVE EMA SMOOTHING
    // ─────────────────────────────────────────────────────────────
    // Prinsip: alpha dihitung dinamis berdasarkan seberapa jauh centroid bergerak.
    //
    //   displacement < MOVE_THRESH  → alpha = ALPHA_STILL  (halus, anti-noise)
    //   displacement > JUMP_THRESH  → alpha = 1.0          (langsung lompat = responsif)
    //   di antara keduanya          → interpolasi linear
    // ─────────────────────────────────────────────────────────────
    private float mSmL = -1f, mSmT = -1f, mSmR = -1f, mSmB = -1f;
    private float mSmCx = -1f, mSmCy = -1f;

    private static final float ALPHA_STILL    = 0.20f;  // alpha saat probe diam
    private static final float ALPHA_FAST     = 1.00f;  // alpha saat probe bergerak cepat
    private static final float MOVE_THRESH_PX = 12f;    // di bawah ini → pakai ALPHA_STILL
    private static final float JUMP_THRESH_PX = 70f;    // di atas ini  → pakai ALPHA_FAST

    // ─────────────────────────────────────────────────────────────
    // TIME-BASED PREDICTION VALIDITY (AUDIT FASE B)
    // ─────────────────────────────────────────────────────────────
    // Menggantikan frame-count hold (MAX_HOLD=6) yang menyebabkan stale detection bertahan 2.5+ detik.
    // Toleransi transien kedip/miss (ms):
    public static final long TIME_HOLD_TOLERANCE_MS = 450L;
    // Waktu maksimal stale sebelum prediksi di-drop permanen (ms):
    public static final long TIME_STALE_MAX_MS      = 750L;

    // ─────────────────────────────────────────────────────────────
    // TEMPORAL HYSTERESIS & SPATIAL GATING (FASE E)
    // ─────────────────────────────────────────────────────────────
    // Berdasarkan audit F1-Score & PR curve empiris di reports/confidence_threshold_analysis.csv:
    public static final float START_THRESHOLD    = 0.55f; // Track baru wajib memiliki confidence tinggi
    public static final float CONTINUE_THRESHOLD = 0.40f; // Track aktif boleh bertahan hingga 0.40
    public static final float SPATIAL_GATE_PX    = 110f;  // Batas toleransi displacement antar fresh prediction

    /** Deteksi terakhir yang valid. */
    private YoloDetector.Detection mLastValid = null;
    private long mLastFreshTimestampNs = 0L;

    // ─────────────────────────────────────────────────────────────
    // ROI KONFIGURASI
    // ─────────────────────────────────────────────────────────────
    // Fokus ke setengah-atas layar: area vena dangkal dekat permukaan kulit.
    private static final float ROI_WIDTH_RATIO  = 0.60f;
    private static final float ROI_HEIGHT_RATIO = 0.45f;
    private static final float ROI_TOP_RATIO    = 0.07f;
    // 1.0x measured median fresh-inference interval on SM-T295 clean replay.
    public static final long MINIMAL_COAST_MS = 344L;

    // =========================================================
    // LIFECYCLE
    // =========================================================

    @Override
    public void onCreate() {
        super.onCreate();
        sInstance    = this;
        mCleanupDone = false;
        mProjectionManager = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);
        mWindowManager     = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
        mMainHandler       = new Handler(getMainLooper());

        DisplayMetrics m = new DisplayMetrics();
        mWindowManager.getDefaultDisplay().getRealMetrics(m);
        mScreenWidth   = m.widthPixels;
        mScreenHeight  = m.heightPixels;
        mScreenDensity = m.densityDpi;
        Log.d(TAG, "onCreate: " + mScreenWidth + "x" + mScreenHeight);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) { stopSelf(); return START_NOT_STICKY; }
        String action = intent.getAction();

        if (ACTION_START.equals(action)) {
            startForegroundWithNotification();

            int    rc  = intent.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED);
            Intent pd  = intent.getParcelableExtra(EXTRA_PROJECTION_DATA);
            if (rc != Activity.RESULT_OK || pd == null) {
                broadcastStopped(); stopForeground(true); stopSelf();
                return START_NOT_STICKY;
            }

            try { mMediaProjection = mProjectionManager.getMediaProjection(rc, pd); }
            catch (Exception e) {
                Log.e(TAG, "getMediaProjection: " + e.getMessage());
                broadcastStopped(); stopForeground(true); stopSelf();
                return START_NOT_STICKY;
            }
            if (mMediaProjection == null) {
                broadcastStopped(); stopForeground(true); stopSelf();
                return START_NOT_STICKY;
            }

            initDetector();
            try { initOverlay(); }   catch (Exception e) { Log.e(TAG, "overlay: " + e); cleanup(); broadcastStopped(); stopForeground(true); stopSelf(); return START_NOT_STICKY; }
            try { startCapture(); }  catch (Exception e) { Log.e(TAG, "capture: " + e); cleanup(); broadcastStopped(); stopForeground(true); stopSelf(); return START_NOT_STICKY; }

            sIsRunning = true;
            broadcastStarted();
            Log.d(TAG, "Service started OK");

        } else if (ACTION_STOP.equals(action)) {
            cleanup(); broadcastStopped(); stopForeground(true); stopSelf();
        }
        return START_NOT_STICKY;
    }

    @Override public IBinder onBind(Intent i) { return null; }

    @Override
    public void onDestroy() {
        super.onDestroy();
        cleanup();
        sInstance  = null;
        sIsRunning = false;
    }

    public static boolean isRunning() { return sIsRunning; }

    public static void setOverlayVisible(boolean visible) {
        VeinDetectionService inst = sInstance;
        if (inst == null || inst.mMainHandler == null) return;
        inst.mMainHandler.post(() -> {
            inst.mOverlayVisible = visible;
            if (inst.mOverlayView != null)
                inst.mOverlayView.setVisibility(visible ? View.VISIBLE : View.INVISIBLE);
        });
    }

    public static void setModeByName(String modeStr) {
        if (modeStr == null) return;
        try {
            if ("LK".equalsIgnoreCase(modeStr) || "LK_EXPERIMENTAL".equalsIgnoreCase(modeStr)) {
                sTemporalMode = TemporalMode.LK_EXPERIMENTAL;
            } else if ("EMA".equalsIgnoreCase(modeStr)) {
                sTemporalMode = TemporalMode.EMA;
            } else if ("RAW".equalsIgnoreCase(modeStr)) {
                sTemporalMode = TemporalMode.RAW;
            } else if ("MINIMAL_ANTIBLINK".equalsIgnoreCase(modeStr)
                    || "ANTIBLINK".equalsIgnoreCase(modeStr)) {
                sTemporalMode = TemporalMode.MINIMAL_ANTIBLINK;
            } else if ("KALMAN".equalsIgnoreCase(modeStr)) {
                sTemporalMode = TemporalMode.KALMAN;
            } else {
                sTemporalMode = TemporalMode.valueOf(modeStr.toUpperCase(java.util.Locale.US));
            }
            Log.i(TAG, "Switched TemporalMode to: " + sTemporalMode);
        } catch (Exception e) {
            Log.e(TAG, "Unknown mode: " + modeStr);
        }
    }

    // =========================================================
    // INIT
    // =========================================================

    private void initDetector() {
        try {
            mYoloDetector = new YoloDetector(this);
            VeinOpticalFlowTracker.initOpenCV();
            mTracker = new VeinOpticalFlowTracker();
            mKalmanTracker = new VeinKalmanTracker();
            mAntiBlinkStateMachine = new AntiBlinkStateMachine(MINIMAL_COAST_MS);
            PerformanceProfiler.getInstance().setModelMetadata(
                    mYoloDetector.getModelName(),
                    mYoloDetector.getInputSize(),
                    mYoloDetector.getDelegateName()
            );
            Log.d(TAG, "YoloDetector, VeinOpticalFlowTracker & VeinKalmanTracker OK (Delegate: " + mYoloDetector.getDelegateName() + ")");
        } catch (Exception e) {
            mYoloDetector = null;
            mTracker = null;
            mKalmanTracker = null;
            Log.e(TAG, "initDetector FAIL: " + e.getMessage());
        }
    }

    private void initOverlay() {
        mOverlayView = new OverlayView(this);
        WindowManager.LayoutParams p = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                        | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT
        );
        mWindowManager.addView(mOverlayView, p);
        mOverlayVisible = true;
        Log.d(TAG, "Overlay added");
    }

    private void startCapture() {
        mProcessingThread = new HandlerThread("VeinCaptureThread");
        mProcessingThread.start();
        mProcessingHandler = new Handler(mProcessingThread.getLooper());

        mFrameSource = new MediaProjectionFrameSource(
                mMediaProjection, mScreenWidth, mScreenHeight, mScreenDensity, mProcessingHandler);
        mFrameSource.start(new FrameSource.Listener() {
            @Override
            public void onFrame(FrameSource.Frame frame) {
                onSourceFrame(frame);
            }

            @Override
            public void onCompleted() {
                mMainHandler.post(() -> {
                    cleanup();
                    broadcastStopped();
                    stopForeground(true);
                    stopSelf();
                });
            }

            @Override
            public void onError(Throwable error) {
                Log.e(TAG, "Frame source failed", error);
            }
        });

        mModeReceiver = new android.content.BroadcastReceiver() {
            @Override
            public void onReceive(Context context, Intent intent) {
                if (intent == null) return;
                String modeStr = intent.getStringExtra("mode");
                if (modeStr != null) {
                    setModeByName(modeStr);
                }
                if (ACTION_CAPTURE_MODEL_INPUT.equals(intent.getAction())) {
                    String captureMode = intent.getStringExtra(EXTRA_CAPTURE_MODE);
                    armModelInputAudit(captureMode);
                }
            }
        };
        android.content.IntentFilter filter = new android.content.IntentFilter("com.example.usgandroidapp.SET_MODE");
        filter.addAction(ACTION_CAPTURE_MODEL_INPUT);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(mModeReceiver, filter, Context.RECEIVER_NOT_EXPORTED);
        } else {
            registerReceiver(mModeReceiver, filter);
        }

        Log.d(TAG, "Screen capture started (TemporalMode: " + sTemporalMode + ")");
    }

    // =========================================================
    // CAMERA THREAD: Latest-Frame Dropping Listener
    // =========================================================

    /**
     * Dipanggil oleh Camera Thread setiap frame tiba.
     * TIDAK memblokir — hanya decode, simpan di slot atomic, trigger AI jika idle.
     */
    private void onSourceFrame(FrameSource.Frame sourceFrame) {
        Bitmap frame = sourceFrame.bitmap;
        try {
            // Inference must not depend on presentation visibility. This also lets Gate 1
            // capture a true no-overlay model input without stopping the detector.
            if (mYoloDetector == null) {
                if (frame != null && !frame.isRecycled()) frame.recycle();
                return;
            }

            if (frame == null) return;
            long captureTs = sourceFrame.captureTimestampNs;

            // 1. Eksekusi Tracker pada frame kamera (~24-30 FPS)
            if (sTemporalMode == TemporalMode.KALMAN) {
                if (mKalmanTracker != null) {
                    VeinKalmanTracker.TrackedVein tracked = mKalmanTracker.onDisplayTick(captureTs, mScreenWidth, mScreenHeight);
                    mMainHandler.postAtFrontOfQueue(() -> {
                        if (mOverlayView != null && mOverlayVisible && sIsRunning) {
                            mOverlayView.setKalmanVein(tracked, mScreenWidth, mScreenHeight);
                        }
                    });
                }
            } else if (sTemporalMode == TemporalMode.LK_EXPERIMENTAL) {
                if (mTracker != null) {
                    VeinOpticalFlowTracker.TrackedVein tracked = mTracker.onCameraFrame(frame, captureTs);
                    if (tracked != null) {
                        mMainHandler.postAtFrontOfQueue(() -> {
                            if (mOverlayView != null && mOverlayVisible && sIsRunning) {
                                mOverlayView.setTrackedVein(tracked, mScreenWidth, mScreenHeight);
                            }
                        });
                    }
                }
            }

            // 2. ATOMIC SWAP: frame baru bersama capture timestamp menggantikan slot YOLO
            FramePacket packet = new FramePacket(frame, captureTs);
            FramePacket discarded = mLatestFrame.getAndSet(packet);
            if (discarded != null && discarded.bitmap != null && !discarded.bitmap.isRecycled()) {
                discarded.bitmap.recycle();
                PerformanceProfiler.getInstance().recordDroppedFrame();
            }

            // Trigger inference hanya jika AI sedang idle
            if (mIsInferring.compareAndSet(false, true)) {
                processLatestFrame();
            }
        } catch (Exception e) {
            if (frame != null && !frame.isRecycled()) frame.recycle();
            Log.e(TAG, "FrameSource listener: " + e.getMessage());
        }
    }

    // =========================================================
    // INFERENCE THREAD: Self-scheduling pipeline
    // =========================================================

    private void processLatestFrame() {
        mInferenceExecutor.execute(() -> {
            FramePacket packet = mLatestFrame.getAndSet(null);
            if (packet == null) { mIsInferring.set(false); return; }
            Bitmap frame = packet.bitmap;
            long captureTimestampNs = packet.captureTimestampNs;

            long tPipelineStart = SystemClock.elapsedRealtimeNanos();

            try {
                // Gate 1: dump before ROI crop, letterbox, preprocessing, or inference.
                // This is deliberately the same Bitmap object passed to detect() below.
                String auditMode = mPendingInputAudit.getAndSet(null);
                if (auditMode != null) {
                    saveExactModelInput(frame, auditMode);
                }

                // ROI: setengah atas, tengah horizontal
                int roiW = (int)(mScreenWidth  * ROI_WIDTH_RATIO);
                int roiH = (int)(mScreenHeight * ROI_HEIGHT_RATIO);
                int roiL = (mScreenWidth - roiW) / 2;
                int roiT = (int)(mScreenHeight  * ROI_TOP_RATIO);

                // Inference with StageTimings and captureTimestampNs
                YoloDetector.StageTimings timings = new YoloDetector.StageTimings();
                List<YoloDetector.Detection> results =
                        mYoloDetector.detect(frame, roiL, roiT, roiW, roiH, timings, captureTimestampNs);

                YoloDetector.Detection best = (sTemporalMode == TemporalMode.RAW
                        || sTemporalMode == TemporalMode.MINIMAL_ANTIBLINK)
                        ? pickHighestConfidence(results) : pickBestWithHysteresis(results);

                // ─────────────────────────────────────────────────────────────
                // TEMPORAL PROCESSING SESUAI MODE AKTIF
                // ─────────────────────────────────────────────────────────────
                if (sTemporalMode == TemporalMode.MINIMAL_ANTIBLINK) {
                    long completedNs = SystemClock.elapsedRealtimeNanos();
                    AntiBlinkStateMachine.Snapshot snapshot = mAntiBlinkStateMachine != null
                            ? mAntiBlinkStateMachine.onFreshInference(best, completedNs) : null;
                    long pipelineNs = SystemClock.elapsedRealtimeNanos() - tPipelineStart;
                    PerformanceProfiler.getInstance().recordInference(
                            timings.preprocessNs, timings.inferenceNs, timings.postprocessNs,
                            timings.maskDecodeNs, 0L, pipelineNs,
                            snapshot != null && snapshot.isVisible(), this);
                    final AntiBlinkStateMachine.Snapshot finalSnapshot = snapshot;
                    mMainHandler.postAtFrontOfQueue(() -> {
                        if (mOverlayView != null && mOverlayVisible && sIsRunning) {
                            mOverlayView.setAntiBlinkSnapshot(finalSnapshot,
                                    mScreenWidth, mScreenHeight);
                        }
                    });

                } else if (sTemporalMode == TemporalMode.KALMAN) {
                    long tKmStart = SystemClock.elapsedRealtimeNanos();
                    VeinKalmanTracker.TrackedVein kResult = null;
                    if (mKalmanTracker != null) {
                        kResult = mKalmanTracker.onYoloMeasurement(best, captureTimestampNs);
                    }
                    long tKmEnd = SystemClock.elapsedRealtimeNanos();
                    long kmNs = tKmEnd - tKmStart;

                    long tPipelineEnd = SystemClock.elapsedRealtimeNanos();
                    long pipelineNs = tPipelineEnd - tPipelineStart;

                    boolean hasDetection = (kResult != null && kResult.state != VeinKalmanTracker.State.LOST);
                    PerformanceProfiler.getInstance().recordInference(
                            timings.preprocessNs,
                            timings.inferenceNs,
                            timings.postprocessNs,
                            timings.maskDecodeNs,
                            kmNs,
                            pipelineNs,
                            hasDetection,
                            this
                    );

                    final VeinKalmanTracker.TrackedVein finalK = kResult;
                    mMainHandler.postAtFrontOfQueue(() -> {
                        if (mOverlayView != null && mOverlayVisible && sIsRunning) {
                            mOverlayView.setKalmanVein(finalK, mScreenWidth, mScreenHeight);
                        }
                    });

                } else if (sTemporalMode == TemporalMode.LK_EXPERIMENTAL) {
                    long tLkStart = SystemClock.elapsedRealtimeNanos();
                    VeinOpticalFlowTracker.TrackedVein corrected = null;
                    if (mTracker != null) {
                        corrected = mTracker.onYoloResult(best, captureTimestampNs);
                    }
                    long tLkEnd = SystemClock.elapsedRealtimeNanos();
                    long lkNs = tLkEnd - tLkStart;

                    long tPipelineEnd = SystemClock.elapsedRealtimeNanos();
                    long pipelineNs = tPipelineEnd - tPipelineStart;

                    boolean hasDetection = (corrected != null && corrected.state != VeinOpticalFlowTracker.TrackState.LOST);
                    PerformanceProfiler.getInstance().recordInference(
                            timings.preprocessNs,
                            timings.inferenceNs,
                            timings.postprocessNs,
                            timings.maskDecodeNs,
                            lkNs,
                            pipelineNs,
                            hasDetection,
                            this
                    );

                    final VeinOpticalFlowTracker.TrackedVein finalLk = corrected;
                    mMainHandler.postAtFrontOfQueue(() -> {
                        if (mOverlayView != null && mOverlayVisible && sIsRunning) {
                            mOverlayView.setTrackedVein(finalLk, mScreenWidth, mScreenHeight);
                        }
                    });

                } else if (sTemporalMode == TemporalMode.EMA) {
                    long tEmaStart = SystemClock.elapsedRealtimeNanos();
                    YoloDetector.Detection toRender = applyEmaAndHold(best, captureTimestampNs);
                    long tEmaEnd = SystemClock.elapsedRealtimeNanos();
                    long emaNs = tEmaEnd - tEmaStart;

                    long tPipelineEnd = SystemClock.elapsedRealtimeNanos();
                    long pipelineNs = tPipelineEnd - tPipelineStart;

                    boolean hasDetection = (toRender != null);
                    PerformanceProfiler.getInstance().recordInference(
                            timings.preprocessNs,
                            timings.inferenceNs,
                            timings.postprocessNs,
                            timings.maskDecodeNs,
                            emaNs,
                            pipelineNs,
                            hasDetection,
                            this
                    );

                    final YoloDetector.Detection finalDet = toRender;
                    mMainHandler.postAtFrontOfQueue(() -> {
                        if (mOverlayView != null && mOverlayVisible && sIsRunning) {
                            mOverlayView.setDetection(finalDet, mScreenWidth, mScreenHeight);
                        }
                    });

                } else { // TemporalMode.RAW
                    long tPipelineEnd = SystemClock.elapsedRealtimeNanos();
                    long pipelineNs = tPipelineEnd - tPipelineStart;

                    boolean hasDetection = (best != null);
                    PerformanceProfiler.getInstance().recordInference(
                            timings.preprocessNs,
                            timings.inferenceNs,
                            timings.postprocessNs,
                            timings.maskDecodeNs,
                            0L,
                            pipelineNs,
                            hasDetection,
                            this
                    );

                    final YoloDetector.Detection finalDet = best;
                    mMainHandler.postAtFrontOfQueue(() -> {
                        if (mOverlayView != null && mOverlayVisible && sIsRunning) {
                            mOverlayView.setRawDetection(finalDet, mScreenWidth, mScreenHeight);
                        }
                    });
                }

            } catch (Exception e) {
                Log.e(TAG, "Inference error: " + e.getMessage());
            } finally {
                if (!frame.isRecycled()) frame.recycle();
                // Self-schedule: jika ada frame baru yang masuk saat kita inferensi, proses segera
                if (mLatestFrame.get() != null) {
                    processLatestFrame();
                } else {
                    mIsInferring.set(false);
                }
            }
        });
    }

    private void armModelInputAudit(String mode) {
        if (mode == null) return;
        final boolean overlayVisible;
        final boolean debugHud;
        switch (mode) {
            case "no_overlay":
                overlayVisible = false;
                debugHud = false;
                break;
            case "with_overlay":
                overlayVisible = true;
                debugHud = false;
                break;
            case "with_hud":
                overlayVisible = true;
                debugHud = true;
                break;
            default:
                Log.e(TAG, "Unknown model-input audit mode: " + mode);
                return;
        }

        mMainHandler.post(() -> {
            mOverlayVisible = overlayVisible;
            OverlayView.sDebugMode = debugHud;
            if (mOverlayView != null) {
                mOverlayView.setVisibility(overlayVisible ? View.VISIBLE : View.INVISIBLE);
                mOverlayView.invalidate();
            }
            // Arm after two display frames so the requested composition state is stable.
            mMainHandler.postDelayed(() -> {
                mPendingInputAudit.set(mode);
                Log.i(TAG, "Armed exact model-input capture: " + mode);
            }, 50L);
        });
    }

    private void saveExactModelInput(Bitmap frame, String mode) {
        FileOutputStream stream = null;
        try {
            File dir = getExternalFilesDir(null);
            if (dir == null) dir = getFilesDir();
            File output = new File(dir, "model_input_" + mode + ".png");
            stream = new FileOutputStream(output, false);
            if (!frame.compress(Bitmap.CompressFormat.PNG, 100, stream)) {
                throw new IllegalStateException("Bitmap.compress returned false");
            }
            stream.flush();
            Log.i(TAG, "Saved exact pre-inference frame: " + output.getAbsolutePath());
        } catch (Exception e) {
            Log.e(TAG, "Failed to save model-input audit frame: " + e.getMessage(), e);
        } finally {
            if (stream != null) {
                try { stream.close(); } catch (Exception ignored) {}
            }
        }
    }

    /**
     * Pilih deteksi terbaik dengan Temporal Hysteresis & Spatial Gating (FASE E).
     *
     * Logika Hysteresis:
     *   1. Jika belum ada track aktif: kandidat WAJIB memiliki confidence >= START_THRESHOLD (0.55).
     *      Noise ber-confidence rendah (29-45%) otomatis ditolak sehingga tidak memicu false track.
     *   2. Jika sudah ada track aktif: kandidat dengan confidence >= CONTINUE_THRESHOLD (0.40)
     *      diterima jika berjarak <= SPATIAL_GATE_PX dari posisi sebelumnya.
     *   3. Jika muncul deteksi baru dengan confidence sangat tinggi (>= START_THRESHOLD),
     *      track boleh berpindah ke lokasi baru tersebut.
     */
    private YoloDetector.Detection pickBestWithHysteresis(List<YoloDetector.Detection> list) {
        if (list == null || list.isEmpty()) return null;

        boolean hasActiveTrack = (mSmL >= 0);
        YoloDetector.Detection best = null;
        float bestConf = -1f;

        for (YoloDetector.Detection cand : list) {
            if (hasActiveTrack) {
                float dx = cand.cx - mSmCx;
                float dy = cand.cy - mSmCy;
                float dist = (float) Math.sqrt(dx * dx + dy * dy);

                // Kompatibel secara spasial dengan track berjalan
                if (dist <= SPATIAL_GATE_PX && cand.confidence >= CONTINUE_THRESHOLD) {
                    if (cand.confidence > bestConf) {
                        bestConf = cand.confidence;
                        best = cand;
                    }
                } else if (cand.confidence >= START_THRESHOLD) {
                    // Deteksi baru di lokasi lain tetapi ber-confidence sangat tinggi
                    if (cand.confidence > bestConf) {
                        bestConf = cand.confidence;
                        best = cand;
                    }
                }
            } else {
                // Belum ada track: wajib lolos START_THRESHOLD
                if (cand.confidence >= START_THRESHOLD) {
                    if (cand.confidence > bestConf) {
                        bestConf = cand.confidence;
                        best = cand;
                    }
                }
            }
        }
        return best;
    }

    private YoloDetector.Detection pickHighestConfidence(List<YoloDetector.Detection> list) {
        if (list == null || list.isEmpty()) return null;
        YoloDetector.Detection best = null;
        for (YoloDetector.Detection candidate : list) {
            if (best == null || candidate.confidence > best.confidence) best = candidate;
        }
        return best;
    }

    /**
     * Terapkan Adaptive EMA pada koordinat bbox & centroid, lalu time-based validity control.
     *
     * ADAPTIVE EMA:
     *   Hitung displacement centroid antara posisi baru vs posisi smooth sekarang.
     *   Jika kecil (probe diam) → alpha rendah → gerakan halus anti-noise.
     *   Jika besar (probe bergerak cepat) → alpha tinggi → overlay langsung mengejar.
     *
     * TIME-BASED VALIDITY (FASE B):
     *   Menggantikan frame-count hold (MAX_HOLD=6) yang menyebabkan stale detection bertahan 2.5+ detik.
     *   Jika deteksi hilang:
     *     - holdAge <= TIME_HOLD_TOLERANCE_MS (450 ms): pertahankan untuk cegah kedip (isHeld=true).
     *     - holdAge <= TIME_STALE_MAX_MS (750 ms): tandai stale (isStale=true, visual decay).
     *     - holdAge > TIME_STALE_MAX_MS: drop permanen (mencegah ghost-vein pada jaringan kosong).
     */
    private YoloDetector.Detection applyEmaAndHold(YoloDetector.Detection raw, long currentTimestampNs) {
        if (raw != null) {
            float disp = 0f;
            float alpha = 1.0f;

            if (mSmL < 0) {
                // Inisialisasi pertama: langsung pakai nilai raw
                mSmL = raw.left;  mSmT = raw.top;
                mSmR = raw.right; mSmB = raw.bottom;
                mSmCx = raw.cx;   mSmCy = raw.cy;
            } else {
                // Hitung displacement centroid → alpha adaptif
                float dx = raw.cx - mSmCx;
                float dy = raw.cy - mSmCy;
                disp  = (float) Math.sqrt(dx * dx + dy * dy);
                float t     = Math.min(1f, Math.max(0f,
                        (disp - MOVE_THRESH_PX) / (JUMP_THRESH_PX - MOVE_THRESH_PX)));
                alpha = ALPHA_STILL + t * (ALPHA_FAST - ALPHA_STILL);

                // EMA update: smooth = smooth + alpha*(target - smooth)
                mSmL  += alpha * (raw.left   - mSmL);
                mSmT  += alpha * (raw.top    - mSmT);
                mSmR  += alpha * (raw.right  - mSmR);
                mSmB  += alpha * (raw.bottom - mSmB);
                mSmCx += alpha * (raw.cx     - mSmCx);
                mSmCy += alpha * (raw.cy     - mSmCy);
            }

            // Terapkan posisi smooth ke objek Detection
            raw.left   = mSmL;  raw.top    = mSmT;
            raw.right  = mSmR;  raw.bottom = mSmB;
            raw.cx     = mSmCx; raw.cy     = mSmCy;
            raw.displacementPx = disp;
            raw.currentAlpha   = alpha;

            raw.isFresh   = true;
            raw.holdAgeMs = 0.0;
            raw.isStale   = false;

            mLastFreshTimestampNs = currentTimestampNs;
            mLastValid = copyDetection(raw);
            return raw;

        } else {
            // Model miss → evaluasi time-based validity (audit FASE B)
            if (mLastValid != null && mLastFreshTimestampNs > 0) {
                long nowNs = SystemClock.elapsedRealtimeNanos();
                double holdAgeMs = (nowNs - mLastFreshTimestampNs) / 1_000_000.0;

                if (holdAgeMs <= TIME_STALE_MAX_MS) {
                    // Masih dalam batas toleransi hold
                    YoloDetector.Detection held = copyDetection(mLastValid);
                    held.isFresh = false;
                    held.holdAgeMs = holdAgeMs;
                    // Ditandai stale jika melewati toleransi transien (450 ms)
                    held.isStale = (holdAgeMs > TIME_HOLD_TOLERANCE_MS);
                    return held;
                } else {
                    // Waktu hold habis (> 750 ms) → segera hapus deteksi basi (anti ghost-vein)
                    mSmL = -1f;
                    mLastValid = null;
                    return null;
                }
            } else {
                mSmL = -1f;
                mLastValid = null;
                return null;
            }
        }
    }

    private YoloDetector.Detection copyDetection(YoloDetector.Detection s) {
        if (s == null) return null;
        YoloDetector.Detection d = new YoloDetector.Detection();
        d.left = s.left; d.top = s.top; d.right = s.right; d.bottom = s.bottom;
        d.cx = s.cx; d.cy = s.cy;
        d.confidence = s.confidence; d.depthMm = s.depthMm;
        d.maskBitmap = s.maskBitmap;
        d.roiLeft = s.roiLeft; d.roiTop = s.roiTop; d.roiWidth = s.roiWidth; d.roiHeight = s.roiHeight;
        d.captureTimestampNs = s.captureTimestampNs;
        d.inferenceEndTimestampNs = s.inferenceEndTimestampNs;
        d.isFresh = s.isFresh; d.holdAgeMs = s.holdAgeMs; d.isStale = s.isStale;
        d.rawLeft = s.rawLeft; d.rawTop = s.rawTop; d.rawRight = s.rawRight; d.rawBottom = s.rawBottom;
        d.rawCx = s.rawCx; d.rawCy = s.rawCy; d.rawConfidence = s.rawConfidence;
        d.displacementPx = s.displacementPx; d.currentAlpha = s.currentAlpha;
        return d;
    }

    // =========================================================
    // CLEANUP
    // =========================================================

    private synchronized void cleanup() {
        if (mCleanupDone) return;
        mCleanupDone = true;
        sIsRunning   = false;

        if (mFrameSource     != null) { try { mFrameSource.stop();        } catch (Exception e) { Log.w(TAG, e); } mFrameSource     = null; }
        if (mMediaProjection != null) { try { mMediaProjection.stop();    } catch (Exception e) { Log.w(TAG, e); } mMediaProjection = null; }

        if (mProcessingThread != null) {
            mProcessingThread.quitSafely();
            mProcessingThread = null; mProcessingHandler = null;
        }
        try { mInferenceExecutor.shutdownNow(); } catch (Exception e) { Log.w(TAG, e); }

        if (mMainHandler != null) {
            mMainHandler.post(() -> {
                if (mOverlayView != null) {
                    try { if (mOverlayView.isAttachedToWindow()) mWindowManager.removeView(mOverlayView); }
                    catch (Exception e) { Log.w(TAG, "removeView: " + e); }
                    mOverlayView = null;
                }
            });
        }
        if (mYoloDetector != null) { try { mYoloDetector.close(); } catch (Exception ignored) {} mYoloDetector = null; }
        if (mTracker != null) { try { mTracker.cleanup(); } catch (Exception ignored) {} mTracker = null; }
        if (mKalmanTracker != null) { try { mKalmanTracker.reset(); } catch (Exception ignored) {} mKalmanTracker = null; }
        if (mAntiBlinkStateMachine != null) { mAntiBlinkStateMachine.reset(); mAntiBlinkStateMachine = null; }
        if (mModeReceiver != null) { try { unregisterReceiver(mModeReceiver); } catch (Exception ignored) {} mModeReceiver = null; }
        PerformanceProfiler.getInstance().dumpReport(this);
        Log.d(TAG, "cleanup() done");
    }

    // =========================================================
    // BROADCAST HELPERS
    // =========================================================

    private void broadcastStarted() { sendBroadcast(new Intent(MainActivity.ACTION_SERVICE_STARTED)); }
    private void broadcastStopped() { sIsRunning = false; sendBroadcast(new Intent(MainActivity.ACTION_SERVICE_STOPPED)); }

    // =========================================================
    // FOREGROUND NOTIFICATION
    // =========================================================

    private void startForegroundWithNotification() {
        NotificationChannel ch = new NotificationChannel(
                NOTIFICATION_CHANNEL_ID, "Deteksi Vena USG", NotificationManager.IMPORTANCE_LOW);
        ch.setDescription("Service deteksi vena aktif");
        ch.setShowBadge(false);
        getSystemService(NotificationManager.class).createNotificationChannel(ch);

        PendingIntent openIntent = PendingIntent.getActivity(this, 0,
                new Intent(this, MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);
        PendingIntent stopIntent = PendingIntent.getService(this, 1,
                new Intent(this, VeinDetectionService.class).setAction(ACTION_STOP),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT);

        Notification notif = new NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
                .setContentTitle("Deteksi Vena Aktif")
                .setContentText("Tap untuk buka app • Tekan STOP untuk berhenti")
                .setSmallIcon(android.R.drawable.ic_menu_view)
                .setContentIntent(openIntent)
                .addAction(android.R.drawable.ic_delete, "STOP", stopIntent)
                .setOngoing(true)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .build();

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(NOTIFICATION_ID, notif, ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION);
        } else {
            startForeground(NOTIFICATION_ID, notif);
        }
    }
}
