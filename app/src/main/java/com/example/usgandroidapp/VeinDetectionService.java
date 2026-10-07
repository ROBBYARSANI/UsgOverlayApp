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
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.media.projection.MediaProjectionManager;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.IBinder;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;

import androidx.core.app.NotificationCompat;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;

public class VeinDetectionService extends Service {

    private static final String TAG = "VeinDetectionService";

    public static final String ACTION_START = "com.example.usgandroidapp.ACTION_START";
    public static final String ACTION_STOP  = "com.example.usgandroidapp.ACTION_STOP";

    public static final String EXTRA_RESULT_CODE     = "extra_result_code";
    public static final String EXTRA_PROJECTION_DATA = "extra_projection_data";

    private static final String NOTIFICATION_CHANNEL_ID = "vein_detection_channel";
    private static final int    NOTIFICATION_ID          = 101;

    private static volatile boolean              sIsRunning = false;
    private static volatile VeinDetectionService sInstance  = null;

    private MediaProjectionManager mProjectionManager;
    private MediaProjection        mMediaProjection;
    private VirtualDisplay         mVirtualDisplay;
    private ImageReader            mImageReader;
    private WindowManager          mWindowManager;
    private OverlayView            mOverlayView;
    private YoloDetector           mYoloDetector;

    private HandlerThread mProcessingThread;
    private Handler       mProcessingHandler;
    private Handler       mMainHandler;

    // Executor terpisah khusus untuk AI
    private final ExecutorService mInferenceExecutor = Executors.newSingleThreadExecutor();

    private int     mScreenWidth;
    private int     mScreenHeight;
    private int     mScreenDensity;
    private boolean mOverlayVisible = true;
    private boolean mCleanupDone    = false;

    // Frame-skipping: jika AI masih memproses frame sebelumnya, frame baru akan dibuang
    private final AtomicBoolean mProcessingFrame = new AtomicBoolean(false);

    // Anti-blink: tahan 2 frame saat model sesaat tidak detect
    private List<YoloDetector.Detection> mLastValidDetections = null;
    private int mHoldFrames = 0;
    private static final int MAX_HOLD_FRAMES = 2;

    // EMA (Exponential Moving Average) smoothing untuk koordinat bbox.
    // Daripada kotak MELOMPAT ke posisi baru, ia MELUNCUR perlahan ke posisi vena.
    // Ini membuat tracking terasa lengket meski FPS hanya 4-5/detik.
    private float mSmoothLeft   = -1f, mSmoothTop    = -1f;
    private float mSmoothRight  = -1f, mSmoothBottom = -1f;
    private float mSmoothCx     = -1f, mSmoothCy     = -1f;
    // Alpha: seberapa cepat kotak "mengejar" posisi terbaru (0.0=tidak bergerak, 1.0=langsung lompat)
    // 0.55 = cukup responsif tapi tidak terlihat melompat
    private static final float EMA_ALPHA = 0.55f;

    @Override
    public void onCreate() {
        super.onCreate();
        sInstance    = this;
        mCleanupDone = false;
        mProjectionManager = (MediaProjectionManager) getSystemService(Context.MEDIA_PROJECTION_SERVICE);
        mWindowManager = (WindowManager) getSystemService(Context.WINDOW_SERVICE);
        mMainHandler   = new Handler(getMainLooper());

        DisplayMetrics metrics = new DisplayMetrics();
        mWindowManager.getDefaultDisplay().getRealMetrics(metrics);
        mScreenWidth   = metrics.widthPixels;
        mScreenHeight  = metrics.heightPixels;
        mScreenDensity = metrics.densityDpi;

        Log.d(TAG, "onCreate: Screen=" + mScreenWidth + "x" + mScreenHeight + " dpi=" + mScreenDensity);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        if (intent == null) {
            Log.w(TAG, "null intent, stopping");
            stopSelf();
            return START_NOT_STICKY;
        }

        String action = intent.getAction();
        Log.d(TAG, "onStartCommand action=" + action);

        if (ACTION_START.equals(action)) {
            startForegroundWithNotification();

            int resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED);
            Intent projData = intent.getParcelableExtra(EXTRA_PROJECTION_DATA);

            if (resultCode != Activity.RESULT_OK || projData == null) {
                Log.e(TAG, "Invalid projection data!");
                broadcastStopped();
                stopForeground(true);
                stopSelf();
                return START_NOT_STICKY;
            }

            try {
                mMediaProjection = mProjectionManager.getMediaProjection(resultCode, projData);
                if (mMediaProjection == null) {
                    Log.e(TAG, "MediaProjection is null!");
                    broadcastStopped();
                    stopForeground(true);
                    stopSelf();
                    return START_NOT_STICKY;
                }
            } catch (Exception e) {
                Log.e(TAG, "getMediaProjection failed: ", e);
                broadcastStopped();
                stopForeground(true);
                stopSelf();
                return START_NOT_STICKY;
            }

            initializeDetector();

            try {
                initializeOverlay();
            } catch (Exception e) {
                Log.e(TAG, "initializeOverlay failed: ", e);
                cleanup();
                broadcastStopped();
                stopForeground(true);
                stopSelf();
                return START_NOT_STICKY;
            }

            try {
                startCapture();
            } catch (Exception e) {
                Log.e(TAG, "startCapture failed: ", e);
                cleanup();
                broadcastStopped();
                stopForeground(true);
                stopSelf();
                return START_NOT_STICKY;
            }

            sIsRunning = true;
            broadcastStarted();
            Log.d(TAG, "Service fully started.");

        } else if (ACTION_STOP.equals(action)) {
            Log.d(TAG, "ACTION_STOP received");
            cleanup();
            broadcastStopped();
            stopForeground(true);
            stopSelf();
        }

        return START_NOT_STICKY;
    }

    @Override
    public IBinder onBind(Intent intent) { return null; }

    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.d(TAG, "onDestroy");
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
            if (inst.mOverlayView != null) {
                inst.mOverlayView.setVisibility(visible ? View.VISIBLE : View.INVISIBLE);
            }
        });
    }

    private void broadcastStarted() {
        sendBroadcast(new Intent(MainActivity.ACTION_SERVICE_STARTED));
    }

    private void broadcastStopped() {
        sIsRunning = false;
        sendBroadcast(new Intent(MainActivity.ACTION_SERVICE_STOPPED));
    }

    private void startForegroundWithNotification() {
        NotificationChannel channel = new NotificationChannel(
                NOTIFICATION_CHANNEL_ID, "Deteksi Vena USG",
                NotificationManager.IMPORTANCE_LOW
        );
        channel.setDescription("Service deteksi vena aktif");
        channel.setShowBadge(false);
        getSystemService(NotificationManager.class).createNotificationChannel(channel);

        PendingIntent openIntent = PendingIntent.getActivity(
                this, 0,
                new Intent(this, MainActivity.class).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT
        );
        PendingIntent stopIntent = PendingIntent.getService(
                this, 1,
                new Intent(this, VeinDetectionService.class).setAction(ACTION_STOP),
                PendingIntent.FLAG_IMMUTABLE | PendingIntent.FLAG_UPDATE_CURRENT
        );

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

    private void initializeDetector() {
        try {
            mYoloDetector = new YoloDetector(this);
            Log.d(TAG, "YoloDetector OK");
        } catch (Exception e) {
            mYoloDetector = null;
            Log.e(TAG, "YoloDetector FAILED: ", e);
        }
    }

    private void initializeOverlay() {
        mOverlayView = new OverlayView(this);
        WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.MATCH_PARENT,
                WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE
                        | WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE
                        | WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT
        );
        mWindowManager.addView(mOverlayView, params);
        mOverlayVisible = true;
        Log.d(TAG, "Overlay added to WindowManager OK");
    }

    private void startCapture() {
        mProcessingThread = new HandlerThread("VeinDetectionThread");
        mProcessingThread.start();
        mProcessingHandler = new Handler(mProcessingThread.getLooper());

        mMediaProjection.registerCallback(new MediaProjection.Callback() {
            @Override
            public void onStop() {
                Log.w(TAG, "MediaProjection onStop() called by system");
                mMainHandler.post(() -> {
                    cleanup();
                    broadcastStopped();
                    stopForeground(true);
                    stopSelf();
                });
            }
        }, mProcessingHandler);

        mImageReader = ImageReader.newInstance(
                mScreenWidth, mScreenHeight,
                PixelFormat.RGBA_8888, 2
        );
        mImageReader.setOnImageAvailableListener(mImageListener, mProcessingHandler);

        mVirtualDisplay = mMediaProjection.createVirtualDisplay(
                "VeinDetectionDisplay",
                mScreenWidth, mScreenHeight, mScreenDensity,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                mImageReader.getSurface(), null, null
        );

        Log.d(TAG, "Screen capture started: " + mScreenWidth + "x" + mScreenHeight);
    }

    private final ImageReader.OnImageAvailableListener mImageListener = reader -> {
        Image image = null;
        try {
            image = reader.acquireLatestImage();
            if (image == null) return;

            // FRAME SKIPPING: Jika AI masih memproses atau detektor tidak tersedia
            if (!mOverlayVisible || mYoloDetector == null || !mProcessingFrame.compareAndSet(false, true)) {
                image.close();
                return;
            }

            Bitmap bitmap = imageToBitmap(image);
            image.close();
            image = null; 

            if (bitmap == null) {
                mProcessingFrame.set(false);
                return;
            }

            // LEPAR KE EXECUTOR TERPISAH agar ImageReader tidak terblokir
            mInferenceExecutor.execute(() -> {
                try {
                    // === ROI: Fokus setengah atas, lebih kecil = lebih cepat ===
                    int roiWidth  = (int) (mScreenWidth  * 0.60f);  // 60% lebar
                    int roiHeight = (int) (mScreenHeight * 0.45f);  // 45% tinggi (setengah atas)
                    int roiLeft   = (mScreenWidth  - roiWidth) / 2;
                    int roiTop    = (int) (mScreenHeight * 0.07f);  // tepat bawah toolbar

                    // Jalankan YOLOv8-seg
                    List<YoloDetector.Detection> yoloResult =
                            mYoloDetector.detect(bitmap, roiLeft, roiTop, roiWidth, roiHeight);

                    List<YoloDetector.Detection> toRender = null;

                    if (yoloResult != null && !yoloResult.isEmpty()) {
                        // Ada deteksi baru — ambil yang paling confident
                        YoloDetector.Detection best = yoloResult.get(0);

                        // === EMA SMOOTHING ===
                        // Jika ini pertama kali, langsung set tanpa smoothing
                        if (mSmoothLeft < 0) {
                            mSmoothLeft   = best.left;
                            mSmoothTop    = best.top;
                            mSmoothRight  = best.right;
                            mSmoothBottom = best.bottom;
                            mSmoothCx     = best.cx;
                            mSmoothCy     = best.cy;
                        } else {
                            // Blend posisi lama dengan posisi baru secara eksponensial.
                            // Kotak MELUNCUR menuju vena, bukan melompat — terasa lengket!
                            mSmoothLeft   = mSmoothLeft   + EMA_ALPHA * (best.left   - mSmoothLeft);
                            mSmoothTop    = mSmoothTop    + EMA_ALPHA * (best.top    - mSmoothTop);
                            mSmoothRight  = mSmoothRight  + EMA_ALPHA * (best.right  - mSmoothRight);
                            mSmoothBottom = mSmoothBottom + EMA_ALPHA * (best.bottom - mSmoothBottom);
                            mSmoothCx     = mSmoothCx     + EMA_ALPHA * (best.cx     - mSmoothCx);
                            mSmoothCy     = mSmoothCy     + EMA_ALPHA * (best.cy     - mSmoothCy);
                        }

                        // Terapkan koordinat yang sudah di-smooth ke detection
                        best.left   = mSmoothLeft;
                        best.top    = mSmoothTop;
                        best.right  = mSmoothRight;
                        best.bottom = mSmoothBottom;
                        best.cx     = mSmoothCx;
                        best.cy     = mSmoothCy;

                        mLastValidDetections = yoloResult;
                        mHoldFrames = MAX_HOLD_FRAMES;
                        toRender = yoloResult;

                    } else {
                        // Tidak ada deteksi — tahan posisi terakhir selama MAX_HOLD_FRAMES
                        if (mHoldFrames > 0 && mLastValidDetections != null) {
                            toRender = mLastValidDetections; // tampilkan posisi smooth terakhir
                            mHoldFrames--;
                        } else {
                            // Benar-benar hilang — reset EMA agar tidak lompat saat muncul lagi
                            mSmoothLeft = -1f;
                            mLastValidDetections = null;
                        }
                    }

                    final List<YoloDetector.Detection> finalResult = toRender;
                    mMainHandler.postAtFrontOfQueue(() -> {
                        if (mOverlayView != null && mOverlayVisible && sIsRunning) {
                            mOverlayView.setDetections(finalResult, mScreenWidth, mScreenHeight);
                        }
                    });
                } finally {
                    if (!bitmap.isRecycled()) bitmap.recycle();
                    mProcessingFrame.set(false);
                }
            });

        } catch (Exception e) {
            mProcessingFrame.set(false);
            Log.e(TAG, "Image processing error: " + e.getMessage());
        } finally {
            if (image != null) image.close();
        }
    };

    private Bitmap imageToBitmap(Image image) {
        try {
            Image.Plane[] planes     = image.getPlanes();
            ByteBuffer    buffer     = planes[0].getBuffer();
            int           pixStride  = planes[0].getPixelStride();
            int           rowStride  = planes[0].getRowStride();
            int           rowPadding = rowStride - pixStride * mScreenWidth;

            Bitmap bmp = Bitmap.createBitmap(
                    mScreenWidth + rowPadding / pixStride,
                    mScreenHeight, Bitmap.Config.ARGB_8888);
            bmp.copyPixelsFromBuffer(buffer);

            if (rowPadding != 0) {
                Bitmap cropped = Bitmap.createBitmap(bmp, 0, 0, mScreenWidth, mScreenHeight);
                bmp.recycle();
                return cropped;
            }
            return bmp;
        } catch (Exception e) {
            Log.e(TAG, "imageToBitmap error: " + e.getMessage());
            return null;
        }
    }

    private synchronized void cleanup() {
        if (mCleanupDone) return;
        mCleanupDone = true;
        sIsRunning   = false;
        Log.d(TAG, "cleanup() start");

        if (mVirtualDisplay  != null) { try { mVirtualDisplay.release();  } catch (Exception e) { Log.w(TAG, e.getMessage()); } mVirtualDisplay  = null; }
        if (mMediaProjection != null) { try { mMediaProjection.stop();    } catch (Exception e) { Log.w(TAG, e.getMessage()); } mMediaProjection = null; }
        if (mImageReader     != null) { try { mImageReader.close();       } catch (Exception e) { Log.w(TAG, e.getMessage()); } mImageReader     = null; }

        if (mProcessingThread != null) {
            mProcessingThread.quitSafely();
            mProcessingThread  = null;
            mProcessingHandler = null;
        }
        try { mInferenceExecutor.shutdownNow(); } catch (Exception e) { Log.w(TAG, e.getMessage()); }

        if (mMainHandler != null) {
            mMainHandler.post(() -> {
                if (mOverlayView != null) {
                    try { if (mOverlayView.isAttachedToWindow()) mWindowManager.removeView(mOverlayView); }
                    catch (Exception e) { Log.w(TAG, "removeView: " + e.getMessage()); }
                    mOverlayView = null;
                }
            });
        }

        if (mYoloDetector != null) { try { mYoloDetector.close(); } catch (Exception ignored) {} mYoloDetector = null; }
        Log.d(TAG, "cleanup() done");
    }
}
