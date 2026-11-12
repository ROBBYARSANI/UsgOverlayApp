package com.example.usgandroidapp;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.graphics.RectF;
import android.os.SystemClock;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Choreographer;
import android.view.View;

import java.util.Locale;

/**
 * OverlayView.java — YOLOv8-Seg Edition
 * =======================================
 * Merender hasil deteksi YOLOv8 di atas layar secara transparan.
 *
 * YANG DIGAMBAR:
 *   1. Mask segmentasi biru semi-transparan (instance mask dari YOLOv8-Seg).
 *   2. Bounding box biru — outline kotak pembatas.
 *   3. Lingkaran merah di centroid (cx, cy).
 *   4. Garis vertikal merah (depth indicator).
 *   5. Label "Conf: X% | Depth: X.X mm".
 *
 * PERSISTENCE OF VISION (60 FPS):
 *   Choreographer memastikan overlay selalu digambar pada refresh rate layar.
 *   AI mengupdate posisi setiap ~50-80ms (320×320 YOLOv8 di Tab A8).
 *   Koordinat tidak pernah blank antar frame → efek "lengket".
 *
 * MASK RENDERING:
 *   Mask 80×80 di-scale ke dalam bbox yang sudah di-smooth via Adaptive EMA.
 *   Dengan menggambar ke bbox (bukan ROI penuh), mask dan bbox SELALU sejajar.
 */
public class OverlayView extends View {

    private static final String TAG = "OverlayView";

    // Overlay mulai memudar setelah ini (ms)
    private static final long FADE_START_MS    = 500L;
    // Durasi fade-out (ms)
    private static final long FADE_DURATION_MS = 300L;

    // ─────────────────────────────────────────────────────────────
    // DEBUG MODE TOGGLE (FASE C)
    // ─────────────────────────────────────────────────────────────
    public static volatile boolean sDebugMode = true;

    // ─────────────────────────────────────────────────────────────
    // PAINT OBJECTS
    // ─────────────────────────────────────────────────────────────
    private final Paint mBoxPaint;
    private final Paint mMaskPaint;
    private final Paint mCirclePaint;
    private final Paint mCircleOuterPaint;
    private final Paint mLinePaint;
    private final Paint mTextPaint;
    private final Paint mTextBgPaint;

    // FASE C: RAW DETECTION & HUD PAINTS
    private final Paint mRawBoxPaint;
    private final Paint mRawCirclePaint;
    private final Paint mDisplacementLinePaint;
    private final Paint mHudPaint;
    private final Paint mHudBgPaint;

    // ─────────────────────────────────────────────────────────────
    // STATE: STICKY (Persistence of Vision)
    // ─────────────────────────────────────────────────────────────
    private float  mStickyL, mStickyT, mStickyR, mStickyB;
    private float  mStickyCx, mStickyCy;
    private float  mStickyConf, mStickyDepthMm;
    private Bitmap mStickyMask;
    private boolean mHasSticky = false;
    private YoloDetector.Detection mCurrentDet = null;
    private VeinOpticalFlowTracker.TrackedVein mCurrentTracked = null;
    private VeinKalmanTracker.TrackedVein mCurrentKalman = null;
    private AntiBlinkStateMachine.Snapshot mCurrentAntiBlink = null;
    private TemporalMode mCurrentMode = TemporalMode.MINIMAL_ANTIBLINK;

    // Visual-only short interpolation. Detector/state-machine coordinates stay authoritative.
    private static final long DISPLAY_INTERPOLATION_MS = 75L;
    private static final float LARGE_CORRECTION_RESET_PX = 110f;
    private float mInterpFromL, mInterpFromT, mInterpFromR, mInterpFromB;
    private float mInterpTargetL, mInterpTargetT, mInterpTargetR, mInterpTargetB;
    private long mInterpStartMs;
    private boolean mInterpolating;

    private long    mLastDetMs = 0L;
    private boolean mChoroRunning = false;

    private int mScreenW = 0;
    private int mScreenH = 0;

    private final Rect  mTextBounds = new Rect();
    private final RectF mMaskDst    = new RectF();

    // ─────────────────────────────────────────────────────────────
    // CHOREOGRAPHER: 60Hz render loop
    // ─────────────────────────────────────────────────────────────
    private final Choreographer.FrameCallback mFrameCb = new Choreographer.FrameCallback() {
        @Override
        public void doFrame(long frameNanos) {
            if (mHasSticky) invalidate();
            if (mChoroRunning) Choreographer.getInstance().postFrameCallback(this);
        }
    };

    // ─────────────────────────────────────────────────────────────
    // API: dipanggil dari VeinDetectionService di Main Thread
    // ─────────────────────────────────────────────────────────────

    /**
     * Update deteksi dari Bounded Constant-Velocity Kalman Tracker (Default Produksi).
     * Dipanggil setiap frame display/kamera (~24-60 FPS) dan saat fresh YOLO tiba.
     */
    public void setKalmanVein(VeinKalmanTracker.TrackedVein vein, int screenWidth, int screenHeight) {
        mScreenW = screenWidth;
        mScreenH = screenHeight;
        mCurrentMode = TemporalMode.KALMAN;
        mCurrentTracked = null;
        mCurrentDet = null;

        if (vein == null || vein.state == VeinKalmanTracker.State.LOST) {
            mHasSticky = false;
            mCurrentKalman = null;
            postInvalidate();
            return;
        }

        mCurrentKalman = vein.copy();
        mStickyL       = vein.bbox.left;
        mStickyT       = vein.bbox.top;
        mStickyR       = vein.bbox.right;
        mStickyB       = vein.bbox.bottom;
        mStickyCx      = vein.cx;
        mStickyCy      = vein.cy;
        mStickyConf    = vein.confidence;
        mStickyDepthMm = vein.depthMm;
        mStickyMask    = vein.maskBitmap;

        mHasSticky = true;
        mLastDetMs = SystemClock.uptimeMillis();
        postInvalidate();
    }

    /**
     * Update deteksi mentah tanpa smoothing temporal (Sample & Hold).
     */
    public void setRawDetection(YoloDetector.Detection det, int screenWidth, int screenHeight) {
        mScreenW = screenWidth;
        mScreenH = screenHeight;
        mCurrentMode = TemporalMode.RAW;
        mCurrentTracked = null;
        mCurrentKalman = null;
        mCurrentDet = det;

        if (det == null) {
            mHasSticky = false;
            postInvalidate();
            return;
        }

        mStickyL       = det.left;
        mStickyT       = det.top;
        mStickyR       = det.right;
        mStickyB       = det.bottom;
        mStickyCx      = det.cx;
        mStickyCy      = det.cy;
        mStickyConf    = det.confidence;
        mStickyDepthMm = det.depthMm;

        if (mStickyMask != null && !mStickyMask.isRecycled() && mStickyMask != det.maskBitmap) {
            mStickyMask.recycle();
        }
        mStickyMask    = det.maskBitmap;

        mHasSticky = true;
        mLastDetMs = SystemClock.uptimeMillis();
        postInvalidate();
    }

    /** Render the minimal SEARCH/LOCKED/COAST/LOST state without altering detector output. */
    public void setAntiBlinkSnapshot(AntiBlinkStateMachine.Snapshot snapshot,
                                     int screenWidth, int screenHeight) {
        mScreenW = screenWidth;
        mScreenH = screenHeight;
        mCurrentMode = TemporalMode.MINIMAL_ANTIBLINK;
        mCurrentKalman = null;
        mCurrentTracked = null;
        mCurrentAntiBlink = snapshot;

        if (snapshot == null || !snapshot.isVisible()
                || snapshot.authoritativeDetection == null) {
            mHasSticky = false;
            mCurrentDet = null;
            mInterpolating = false;
            if (mStickyMask != null && !mStickyMask.isRecycled()) mStickyMask.recycle();
            mStickyMask = null;
            postInvalidate();
            return;
        }

        YoloDetector.Detection det = snapshot.authoritativeDetection;
        mCurrentDet = det;
        if (!mHasSticky) {
            setDisplayedBox(det.left, det.top, det.right, det.bottom);
            mInterpolating = false;
        } else if (snapshot.freshResult) {
            updateInterpolatedBox(SystemClock.uptimeMillis());
            float oldCx = (mStickyL + mStickyR) * 0.5f;
            float oldCy = (mStickyT + mStickyB) * 0.5f;
            float jump = (float) Math.hypot(det.cx - oldCx, det.cy - oldCy);
            if (jump > LARGE_CORRECTION_RESET_PX) {
                setDisplayedBox(det.left, det.top, det.right, det.bottom);
                mInterpolating = false;
            } else {
                mInterpFromL = mStickyL; mInterpFromT = mStickyT;
                mInterpFromR = mStickyR; mInterpFromB = mStickyB;
                mInterpTargetL = det.left; mInterpTargetT = det.top;
                mInterpTargetR = det.right; mInterpTargetB = det.bottom;
                mInterpStartMs = SystemClock.uptimeMillis();
                mInterpolating = true;
            }
        }

        mStickyConf = det.confidence;
        mStickyDepthMm = det.depthMm;
        if (snapshot.freshResult && mStickyMask != det.maskBitmap) {
            if (mStickyMask != null && !mStickyMask.isRecycled()) mStickyMask.recycle();
            mStickyMask = det.maskBitmap;
        }
        mHasSticky = true;
        mLastDetMs = SystemClock.uptimeMillis();
        postInvalidate();
    }

    private void setDisplayedBox(float left, float top, float right, float bottom) {
        mStickyL = left; mStickyT = top; mStickyR = right; mStickyB = bottom;
        mStickyCx = (left + right) * 0.5f;
        mStickyCy = (top + bottom) * 0.5f;
    }

    private void updateInterpolatedBox(long nowMs) {
        if (!mInterpolating) return;
        float progress = Math.min(1f, Math.max(0f,
                (nowMs - mInterpStartMs) / (float) DISPLAY_INTERPOLATION_MS));
        float eased = progress * progress * (3f - 2f * progress);
        setDisplayedBox(
                mInterpFromL + (mInterpTargetL - mInterpFromL) * eased,
                mInterpFromT + (mInterpTargetT - mInterpFromT) * eased,
                mInterpFromR + (mInterpTargetR - mInterpFromR) * eased,
                mInterpFromB + (mInterpTargetB - mInterpFromB) * eased);
        if (progress >= 1f) mInterpolating = false;
    }

    /**
     * Update deteksi dari Continuous Hybrid Tracker (Lucas-Kanade Optical Flow + YOLOv8-Seg).
     * Mode Eksperimental (LK_EXPERIMENTAL).
     */
    public void setTrackedVein(VeinOpticalFlowTracker.TrackedVein tracked, int screenWidth, int screenHeight) {
        mScreenW = screenWidth;
        mScreenH = screenHeight;
        mCurrentMode = TemporalMode.LK_EXPERIMENTAL;
        mCurrentKalman = null;
        mCurrentDet = null;

        if (tracked == null || tracked.state == VeinOpticalFlowTracker.TrackState.LOST) {
            mHasSticky = false;
            mCurrentTracked = null;
            postInvalidate();
            return;
        }

        mCurrentTracked = tracked.copy();
        mStickyL       = tracked.bbox.left;
        mStickyT       = tracked.bbox.top;
        mStickyR       = tracked.bbox.right;
        mStickyB       = tracked.bbox.bottom;
        mStickyCx      = tracked.cx;
        mStickyCy      = tracked.cy;
        mStickyConf    = tracked.confidence;
        mStickyDepthMm = tracked.depthMm;
        mStickyMask    = tracked.maskBitmap;

        mHasSticky = true;
        mLastDetMs = SystemClock.uptimeMillis();
        postInvalidate();
    }

    /**
     * Update deteksi terbaru (Legacy YOLOv8/EMA fallback).
     */
    public void setDetection(YoloDetector.Detection det, int screenWidth, int screenHeight) {
        mScreenW = screenWidth;
        mScreenH = screenHeight;
        mCurrentMode = TemporalMode.EMA;
        mCurrentKalman = null;
        mCurrentTracked = null;
        mCurrentDet = det;

        if (det != null) {
            mStickyL      = det.left;
            mStickyT      = det.top;
            mStickyR      = det.right;
            mStickyB      = det.bottom;
            mStickyCx     = det.cx;
            mStickyCy     = det.cy;
            mStickyConf   = det.confidence;
            mStickyDepthMm= det.depthMm;

            if (mStickyMask != null && !mStickyMask.isRecycled() && mStickyMask != det.maskBitmap) {
                mStickyMask.recycle();
            }
            mStickyMask = det.maskBitmap;

            mHasSticky = true;
            mLastDetMs = SystemClock.uptimeMillis();
            postInvalidate();
        }
    }

    // ─────────────────────────────────────────────────────────────
    // ONDRAW — dipanggil 60x/detik oleh Choreographer
    // ─────────────────────────────────────────────────────────────

    @Override
    protected void onDraw(Canvas canvas) {
        long tStart = SystemClock.elapsedRealtimeNanos();
        super.onDraw(canvas);

        if (!mHasSticky) {
            // Jika state LOST tetapi debug HUD aktif, gambarkan compact HUD LOST di pojok kiri bawah
            if (sDebugMode) {
                drawCompactDebugHud(canvas);
            }
            long tEnd = SystemClock.elapsedRealtimeNanos();
            PerformanceProfiler.getInstance().recordOnDraw(tEnd - tStart);
            return;
        }

        long drawNs = SystemClock.elapsedRealtimeNanos();
        if (mCurrentMode == TemporalMode.MINIMAL_ANTIBLINK) {
            updateInterpolatedBox(SystemClock.uptimeMillis());
        }
        long captureNs = 0L;
        if (mCurrentKalman != null && mCurrentKalman.captureTimestampNs > 0) {
            captureNs = mCurrentKalman.captureTimestampNs;
        } else if (mCurrentTracked != null && mCurrentTracked.captureTimestampNs > 0) {
            captureNs = mCurrentTracked.captureTimestampNs;
        } else if (mCurrentDet != null && mCurrentDet.captureTimestampNs > 0) {
            captureNs = mCurrentDet.captureTimestampNs;
        }
        double predAgeMs = (captureNs > 0) ? (drawNs - captureNs) / 1_000_000.0 : 0.0;

        // ─────────────────────────────────────────────────────────────
        // PHASE 2 & 4 — FLICKER-FREE SOLID STATE MACHINE
        // ─────────────────────────────────────────────────────────────
        int stateColor;
        if (mCurrentMode == TemporalMode.MINIMAL_ANTIBLINK) {
            // LOCKED and COAST are intentionally pixel-identical. State is HUD-only.
            stateColor = Color.rgb(0, 240, 255);
        } else if (mCurrentMode == TemporalMode.KALMAN && mCurrentKalman != null) {
            switch (mCurrentKalman.state) {
                case FRESH:
                    stateColor = Color.rgb(0, 240, 255); // Cyan solid (Fresh YOLO confirmation)
                    break;
                case PREDICTED:
                    stateColor = Color.rgb(0, 229, 255); // Teal solid (Inter-frame Kalman velocity)
                    break;
                case GRACE:
                    stateColor = Color.rgb(255, 190, 40); // Amber solid (Anti-blink grace window)
                    break;
                case LOST:
                default:
                    mHasSticky = false;
                    mCurrentKalman = null;
                    if (sDebugMode) drawCompactDebugHud(canvas);
                    long tEnd = SystemClock.elapsedRealtimeNanos();
                    PerformanceProfiler.getInstance().recordOnDraw(tEnd - tStart);
                    return;
            }
        } else if (mCurrentMode == TemporalMode.LK_EXPERIMENTAL && mCurrentTracked != null) {
            switch (mCurrentTracked.state) {
                case FRESH_YOLO:
                    stateColor = Color.rgb(0, 240, 255);
                    break;
                case TRACKED:
                    stateColor = Color.rgb(0, 229, 255);
                    break;
                case DEGRADED:
                    stateColor = Color.rgb(255, 179, 0);
                    break;
                case LOST:
                default:
                    mHasSticky = false;
                    if (sDebugMode) drawCompactDebugHud(canvas);
                    long tEnd = SystemClock.elapsedRealtimeNanos();
                    PerformanceProfiler.getInstance().recordOnDraw(tEnd - tStart);
                    return;
            }
        } else {
            // RAW / EMA fallback
            stateColor = (mCurrentDet != null && mCurrentDet.isStale)
                    ? Color.rgb(255, 179, 0) : Color.rgb(0, 240, 255);
        }

        mBoxPaint.setColor(stateColor);
        mBoxPaint.setAlpha(255); // Solid stroke, NO blinking, NO opacity pulsation
        mCirclePaint.setColor(stateColor);
        mCirclePaint.setAlpha(255);

        // 1. Mask segmentasi instance (diposisikan presisi ke dalam bbox)
        drawMask(canvas, 180);

        // 2. Bounding box dengan kontras tinggi (Solid stroke)
        canvas.drawRect(mStickyL, mStickyT, mStickyR, mStickyB, mBoxPaint);

        // 3. Depth indicator: garis vertikal tegas dari atas bbox ke centroid
        canvas.drawLine(mStickyCx, mStickyT, mStickyCx, mStickyCy, mLinePaint);

        // 4. Centroid reticle: outer ring putih + titik solid
        canvas.drawCircle(mStickyCx, mStickyCy, 9f, mCircleOuterPaint);
        canvas.drawCircle(mStickyCx, mStickyCy, 5f, mCirclePaint);

        // 5. Label info diletakkan di LUAR bbox agar tidak menutupi lumen vena
        String label = String.format(Locale.US,
                "Conf: %.0f%%  Depth: %.1f mm", mStickyConf * 100f, mStickyDepthMm);
        float labelY = (mStickyT > 55f) ? (mStickyT - 12f) : (mStickyB + 36f);
        drawLabel(canvas, label, mStickyL, labelY);

        // 6. PHASE 9 — COMPACT DEBUG HUD (Bottom-Left, safe margin 55dp)
        if (sDebugMode) {
            drawCompactDebugHud(canvas);
        }

        // 7. RECORD TELEMETRY SAMPLES
        if (mCurrentMode == TemporalMode.KALMAN && mCurrentKalman != null) {
            PerformanceProfiler.getInstance().recordKalmanSample(
                    captureNs,
                    mCurrentKalman.state.name(),
                    (float) predAgeMs,
                    mStickyConf,
                    mStickyCx,
                    mStickyCy,
                    mStickyR - mStickyL,
                    mStickyB - mStickyT,
                    true,
                    0f,
                    false,
                    getContext()
            );
        } else if (mCurrentMode == TemporalMode.LK_EXPERIMENTAL && mCurrentTracked != null) {
            PerformanceProfiler.getInstance().recordHybridTrackingSample(
                    mCurrentTracked.captureTimestampNs,
                    mCurrentTracked.isFreshYolo ? "YOLO_CORRECTION" : "OPTICAL_FLOW",
                    mCurrentTracked.state.name(),
                    (float) predAgeMs,
                    mCurrentTracked.rawConfidence,
                    mCurrentTracked.confidence,
                    mCurrentTracked.numInputPoints,
                    mCurrentTracked.numValidPoints,
                    mCurrentTracked.fbErrorPx,
                    mCurrentTracked.lkErrorPx,
                    0f, 0f,
                    mStickyCx, mStickyCy,
                    0f, 0f,
                    false,
                    getContext()
            );
        } else if (mCurrentDet != null) {
            PerformanceProfiler.getInstance().recordTemporalSample(
                    mCurrentDet.captureTimestampNs,
                    mCurrentDet.inferenceEndTimestampNs,
                    drawNs,
                    predAgeMs,
                    mCurrentDet.isFresh,
                    !mCurrentDet.isFresh,
                    mCurrentDet.holdAgeMs,
                    mCurrentDet.isStale,
                    mCurrentDet.rawConfidence,
                    mCurrentDet.rawCx, mCurrentDet.rawCy,
                    mStickyCx, mStickyCy,
                    mCurrentDet.rawLeft, mCurrentDet.rawTop, mCurrentDet.rawRight, mCurrentDet.rawBottom,
                    mStickyL, mStickyT, mStickyR, mStickyB,
                    true,
                    getContext()
            );
        }

        long tEnd = SystemClock.elapsedRealtimeNanos();
        PerformanceProfiler.getInstance().recordOnDraw(tEnd - tStart);
    }

    /**
     * Compact Debug HUD (Phase 9).
     * Posisi: BOTTOM-LEFT, di atas toolbar/control area USG.
     * Ukuran target: width ~160dp, height ~44dp.
     * Layout maksimal 2 baris:
     *   Line 1: [STATE]  age:XXms  conf:.XX
     *   Line 2: mode:[MODE]  status:[INFO]
     */
    private void drawCompactDebugHud(Canvas canvas) {
        float density = getResources().getDisplayMetrics().density;
        float scaledDensity = getResources().getDisplayMetrics().scaledDensity;

        float hudW = 175f * density;
        float hudH = 46f * density;
        float hudX = 8f * density;
        // Bottom safe margin: 55dp di atas batas bawah layar
        float hudY = mScreenH - (55f * density) - hudH;
        if (hudY < 0) hudY = 10f;

        // Rounded semi-transparent black background
        RectF hudRect = new RectF(hudX, hudY, hudX + hudW, hudY + hudH);
        canvas.drawRoundRect(hudRect, 6f * density, 6f * density, mHudBgPaint);

        String stateStr = "LOST";
        int stateColor = Color.GRAY;
        float predAge = 0f;
        float conf = 0f;
        String line2Str;

        if (mCurrentMode == TemporalMode.MINIMAL_ANTIBLINK && mCurrentAntiBlink != null) {
            stateStr = mCurrentAntiBlink.state.name();
            stateColor = mCurrentAntiBlink.state == AntiBlinkStateMachine.State.LOCKED
                    ? Color.GREEN : (mCurrentAntiBlink.state == AntiBlinkStateMachine.State.COAST
                    ? Color.YELLOW : Color.GRAY);
            predAge = (float) mCurrentAntiBlink.ageMs;
            conf = mCurrentDet != null ? mCurrentDet.confidence : 0f;
            line2Str = "MINIMAL coast:344ms";
        } else if (mCurrentMode == TemporalMode.KALMAN && mCurrentKalman != null) {
            switch (mCurrentKalman.state) {
                case FRESH:
                    stateStr = "FRESH";
                    stateColor = Color.rgb(0, 240, 255);
                    break;
                case PREDICTED:
                    stateStr = "PRED";
                    stateColor = Color.rgb(0, 229, 255);
                    break;
                case GRACE:
                    stateStr = "GRACE";
                    stateColor = Color.rgb(255, 190, 40);
                    break;
                case LOST:
                default:
                    stateStr = "LOST";
                    stateColor = Color.GRAY;
                    break;
            }
            predAge = mCurrentKalman.predAgeMs;
            conf = mCurrentKalman.confidence;
            line2Str = "KALMAN  horizon:600ms";

        } else if (mCurrentMode == TemporalMode.LK_EXPERIMENTAL && mCurrentTracked != null) {
            switch (mCurrentTracked.state) {
                case FRESH_YOLO:
                    stateStr = "FRESH";
                    stateColor = Color.rgb(0, 240, 255);
                    break;
                case TRACKED:
                    stateStr = "TRACK";
                    stateColor = Color.rgb(0, 229, 255);
                    break;
                case DEGRADED:
                    stateStr = "DEGRD";
                    stateColor = Color.rgb(255, 179, 0);
                    break;
                case LOST:
                default:
                    stateStr = "LOST";
                    stateColor = Color.GRAY;
                    break;
            }
            predAge = mCurrentTracked.predAgeMs;
            conf = mCurrentTracked.confidence;
            line2Str = String.format(Locale.US, "pts:%-2d  fb:%.1fpx",
                    mCurrentTracked.numValidPoints, mCurrentTracked.fbErrorPx);

        } else if (mCurrentDet != null) {
            if (mCurrentMode == TemporalMode.RAW) {
                stateStr = "RAW";
                stateColor = Color.rgb(0, 240, 255);
                line2Str = "RAW YOLO  sample-hold";
            } else {
                stateStr = mCurrentDet.isFresh ? "FRESH" : (mCurrentDet.isStale ? "STALE" : "HELD");
                stateColor = mCurrentDet.isFresh ? Color.GREEN : (mCurrentDet.isStale ? Color.RED : Color.YELLOW);
                line2Str = String.format(Locale.US, "EMA  a:%.2f", mCurrentDet.currentAlpha);
            }
            conf = mCurrentDet.confidence;
        } else {
            stateStr = "LOST";
            stateColor = Color.GRAY;
            line2Str = "mode:" + mCurrentMode.name();
        }

        mHudPaint.setTextSize(10f * scaledDensity);

        // Baris 1: STATE  age:XXms  conf:.XX
        float line1Y = hudY + 18f * density;
        mHudPaint.setColor(stateColor);
        canvas.drawText(stateStr, hudX + 8f * density, line1Y, mHudPaint);

        mHudPaint.setColor(Color.WHITE);
        String ageConfStr = String.format(Locale.US, "  age:%.0fms  C:.%02d",
                predAge, (int)(conf * 100));
        float stateW = mHudPaint.measureText(stateStr);
        canvas.drawText(ageConfStr, hudX + 8f * density + stateW, line1Y, mHudPaint);

        // Baris 2: info status mode
        float line2Y = hudY + 36f * density;
        mHudPaint.setColor(Color.LTGRAY);
        canvas.drawText(line2Str, hudX + 8f * density, line2Y, mHudPaint);
    }

    /**
     * Gambar mask instance ke dalam bbox yang sudah di-smooth.
     *
     * KRITIS: Mask selalu digambar ke dst = bbox (bukan ROI penuh).
     * Karena koordinat bbox sudah di-smooth via Adaptive EMA di VeinDetectionService,
     * dan mask bitmap dari frame terbaru, keduanya akan selalu sejajar.
     * Tidak ada lagi kasus mask "keluar dari bbox".
     */
    private void drawMask(Canvas canvas, int alpha) {
        if (mStickyMask == null || mStickyMask.isRecycled()) return;

        // Dst rect = bbox (BUKAN ROI penuh yang fixed)
        mMaskDst.set(mStickyL, mStickyT, mStickyR, mStickyB);
        mMaskPaint.setAlpha((int)(alpha * 0.85f));
        canvas.drawBitmap(mStickyMask, null, mMaskDst, mMaskPaint);
    }

    private int computeAlpha(long elapsedMs) {
        if (elapsedMs < FADE_START_MS) return 255;
        long fadeMs = elapsedMs - FADE_START_MS;
        if (fadeMs >= FADE_DURATION_MS) return 0;
        return (int)(255 * (1f - (float) fadeMs / FADE_DURATION_MS));
    }

    private void applyAlpha(int alpha) {
        mBoxPaint.setAlpha(alpha);
        mCirclePaint.setAlpha(alpha);
        mCircleOuterPaint.setAlpha(alpha);
        mLinePaint.setAlpha(alpha);
        mTextPaint.setAlpha(alpha);
        mTextBgPaint.setAlpha((int)(alpha * 0.78f));
    }

    private void drawLabel(Canvas canvas, String text, float x, float y) {
        mTextPaint.getTextBounds(text, 0, text.length(), mTextBounds);
        float tw = mTextBounds.width(), th = mTextBounds.height(), pad = 6f;
        canvas.drawRect(x - pad, y - th - pad, x + tw + pad, y + pad, mTextBgPaint);
        canvas.drawText(text, x, y, mTextPaint);
    }

    // ─────────────────────────────────────────────────────────────
    // LIFECYCLE
    // ─────────────────────────────────────────────────────────────

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        mChoroRunning = true;
        Choreographer.getInstance().postFrameCallback(mFrameCb);
        Log.d(TAG, "Choreographer 60Hz started");
    }

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        mChoroRunning = false;
        Choreographer.getInstance().removeFrameCallback(mFrameCb);
        // Cleanup sticky mask
        if (mStickyMask != null && !mStickyMask.isRecycled()) {
            mStickyMask.recycle();
            mStickyMask = null;
        }
        Log.d(TAG, "Choreographer stopped");
    }

    // ─────────────────────────────────────────────────────────────
    // CONSTRUCTORS
    // ─────────────────────────────────────────────────────────────

    public OverlayView(Context context) {
        super(context);
        mBoxPaint              = makeBoxPaint();
        mMaskPaint             = makeMaskPaint();
        mCirclePaint           = makeCirclePaint();
        mCircleOuterPaint      = makeCircleOuterPaint();
        mLinePaint             = makeLinePaint();
        mTextPaint             = makeTextPaint();
        mTextBgPaint           = makeTextBgPaint();
        mRawBoxPaint           = makeRawBoxPaint();
        mRawCirclePaint        = makeRawCirclePaint();
        mDisplacementLinePaint = makeDisplacementLinePaint();
        mHudPaint              = makeHudPaint();
        mHudBgPaint            = makeHudBgPaint();
    }

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        mBoxPaint              = makeBoxPaint();
        mMaskPaint             = makeMaskPaint();
        mCirclePaint           = makeCirclePaint();
        mCircleOuterPaint      = makeCircleOuterPaint();
        mLinePaint             = makeLinePaint();
        mTextPaint             = makeTextPaint();
        mTextBgPaint           = makeTextBgPaint();
        mRawBoxPaint           = makeRawBoxPaint();
        mRawCirclePaint        = makeRawCirclePaint();
        mDisplacementLinePaint = makeDisplacementLinePaint();
        mHudPaint              = makeHudPaint();
        mHudBgPaint            = makeHudBgPaint();
    }

    // ─────────────────────────────────────────────────────────────
    // PAINT FACTORIES
    // ─────────────────────────────────────────────────────────────

    private Paint makeBoxPaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.argb(255, 0, 240, 255)); // Bright high-contrast Cyan
        p.setStyle(Paint.Style.STROKE);
        p.setStrokeWidth(4.5f);
        return p;
    }

    private Paint makeMaskPaint() {
        Paint p = new Paint();
        p.setFlags(Paint.FILTER_BITMAP_FLAG); // bilinear filter saat scale
        return p;
    }

    private Paint makeCirclePaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.RED);
        p.setStyle(Paint.Style.FILL);
        return p;
    }

    private Paint makeCircleOuterPaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.WHITE);
        p.setStyle(Paint.Style.STROKE);
        p.setStrokeWidth(2.5f);
        return p;
    }

    private Paint makeLinePaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.argb(220, 255, 235, 59)); // Bright Yellow
        p.setStyle(Paint.Style.STROKE);
        p.setStrokeWidth(2.5f);
        return p;
    }

    private Paint makeTextPaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.WHITE);
        p.setTextSize(30f);
        return p;
    }

    private Paint makeTextBgPaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.argb(200, 0, 0, 0));
        p.setStyle(Paint.Style.FILL);
        return p;
    }

    private Paint makeRawBoxPaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.argb(255, 255, 235, 59)); // Bright Yellow (FASE C: RAW YOLO)
        p.setStyle(Paint.Style.STROKE);
        p.setStrokeWidth(3.0f);
        return p;
    }

    private Paint makeRawCirclePaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.argb(255, 255, 235, 59)); // Yellow circle
        p.setStyle(Paint.Style.FILL);
        return p;
    }

    private Paint makeDisplacementLinePaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.argb(220, 255, 255, 255)); // White vector line
        p.setStyle(Paint.Style.STROKE);
        p.setStrokeWidth(2.0f);
        return p;
    }

    private Paint makeHudPaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.WHITE);
        p.setTextSize(26f);
        p.setTypeface(android.graphics.Typeface.MONOSPACE);
        return p;
    }

    private Paint makeHudBgPaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.argb(210, 20, 20, 20));
        p.setStyle(Paint.Style.FILL);
        return p;
    }
}
