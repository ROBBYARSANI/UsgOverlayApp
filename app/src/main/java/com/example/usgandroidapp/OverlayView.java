package com.example.usgandroidapp;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.util.AttributeSet;
import android.util.Log;
import android.view.Choreographer;
import android.view.View;

import java.util.List;
import java.util.Locale;

/**
 * OverlayView.java — Persistence of Vision Renderer
 * ====================================================
 * Terinspirasi dari lampu neon yang berkedip 100-120x/detik sehingga
 * mata manusia melihatnya sebagai cahaya KONSTAN (persistence of vision).
 *
 * Prinsip yang sama diterapkan di sini:
 * - Layar tablet refresh 60Hz (setiap 16ms)
 * - Model AI update posisi setiap ~50-80ms
 * - OverlayView SELALU menggambar bbox di posisi terakhir yang diketahui
 *   pada SETIAP frame layar — bbox tidak pernah menghilang karena model "miss"
 * - Bbox hanya memudar perlahan (fade-out) jika model benar-benar tidak
 *   mendeteksi apapun selama >400ms
 *
 * Hasilnya: bounding box terasa LENGKET dan mengikuti vena secara mulus.
 */
public class OverlayView extends View {

    private static final String TAG = "OverlayView";

    // Durasi (ms) sebelum bbox mulai memudar jika tidak ada deteksi baru.
    // 400ms = 5-8 siklus inferensi 320x320 — artinya model benar-benar hilang.
    private static final long FADE_START_MS = 400L;
    // Durasi total fade-out dari alpha penuh ke 0
    private static final long FADE_DURATION_MS = 300L;

    // =========================================================
    // PAINT OBJECTS
    // =========================================================
    private final Paint mBoxPaint;
    private final Paint mCirclePaint;
    private final Paint mLinePaint;
    private final Paint mTextPaint;
    private final Paint mTextBgPaint;

    // =========================================================
    // STATE: PERSISTENCE OF VISION
    // =========================================================
    // Bbox terakhir yang valid — TIDAK pernah dihapus saat model "miss"
    private float mStickyLeft, mStickyTop, mStickyRight, mStickyBottom;
    private float mStickyCx, mStickyCy;
    private float mStickyDepthMm;
    private boolean mHasSticky = false; // apakah ada bbox yang pernah terdeteksi

    // Timestamp (SystemClock.uptimeMillis) kapan deteksi terakhir diterima
    private long mLastDetectionTimeMs = 0L;

    // Apakah Choreographer loop sedang aktif
    private boolean mChoreographerRunning = false;

    private int mScreenWidth  = 0;
    private int mScreenHeight = 0;

    private final Rect mTextBounds = new Rect();

    // =========================================================
    // CHOREOGRAPHER: 60Hz RENDER LOOP
    // =========================================================
    // Dipanggil setiap frame layar (~16ms) — seperti lampu neon yang
    // terus-menerus menyala tanpa terlihat berkedip.
    private final Choreographer.FrameCallback mFrameCallback = new Choreographer.FrameCallback() {
        @Override
        public void doFrame(long frameTimeNanos) {
            if (mHasSticky) {
                // Gambar ulang setiap frame layar — ini yang menciptakan
                // persistence of vision, bbox tidak pernah "blank" antar frame!
                invalidate();
            }
            if (mChoreographerRunning) {
                Choreographer.getInstance().postFrameCallback(this);
            }
        }
    };

    // =========================================================
    // API: DIPANGGIL DARI VeinDetectionService
    // =========================================================

    /**
     * Dipanggil oleh service setiap ~50-80ms (kecepatan model 320x320).
     * Jika detections kosong/null, bbox TIDAK dihapus — hanya waktu "terakhir
     * terlihat" yang tidak diperbarui, lalu bbox perlahan memudar.
     */
    public void setDetections(List<YoloDetector.Detection> newDetections,
                              int screenWidth, int screenHeight) {
        this.mScreenWidth  = screenWidth;
        this.mScreenHeight = screenHeight;

        if (newDetections != null && !newDetections.isEmpty()) {
            // Ada deteksi baru — perbarui posisi sticky
            YoloDetector.Detection best = newDetections.get(0);
            mStickyLeft    = best.left;
            mStickyTop     = best.top;
            mStickyRight   = best.right;
            mStickyBottom  = best.bottom;
            mStickyCx      = best.cx;
            mStickyCy      = best.cy;
            mStickyDepthMm = best.depthMm;
            mHasSticky     = true;
            mLastDetectionTimeMs = android.os.SystemClock.uptimeMillis();
        }
        // Jika kosong: TIDAK reset mHasSticky — bbox tetap tergambar oleh Choreographer!
    }

    // =========================================================
    // ONDRAW — Dipanggil 60x per detik oleh Choreographer
    // =========================================================

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        if (!mHasSticky) return;

        // Hitung alpha berdasarkan waktu sejak deteksi terakhir
        long elapsedMs = android.os.SystemClock.uptimeMillis() - mLastDetectionTimeMs;
        int alpha = computeAlpha(elapsedMs);

        if (alpha <= 0) {
            // Sudah memudar total — reset agar tidak terus render kosong
            mHasSticky = false;
            return;
        }

        // Terapkan alpha ke semua Paint — semakin lama tidak terdeteksi, semakin transparan
        applyAlpha(alpha);

        // Gambar bbox di posisi terakhir yang diketahui
        canvas.drawRect(mStickyLeft, mStickyTop, mStickyRight, mStickyBottom, mBoxPaint);
        canvas.drawLine(mStickyCx, mStickyTop, mStickyCx, mStickyCy, mLinePaint);
        canvas.drawCircle(mStickyCx, mStickyCy, 12.0f, mCirclePaint);

        // Label depth hanya tampil saat alpha cukup tinggi (tidak saat memudar)
        if (alpha > 100) {
            String label = String.format(Locale.US, "Depth: %.1f mm", mStickyDepthMm);
            drawLabelWithBackground(canvas, label, mStickyCx + 20, mStickyCy, alpha);
        }
    }

    /**
     * Hitung alpha (0-255) berdasarkan waktu sejak deteksi terakhir.
     * Penuh (255) selama FADE_START_MS, lalu turun linear ke 0 selama FADE_DURATION_MS.
     */
    private int computeAlpha(long elapsedMs) {
        if (elapsedMs < FADE_START_MS) {
            return 255; // Deteksi masih "segar" — tampil penuh
        }
        long fadeElapsed = elapsedMs - FADE_START_MS;
        if (fadeElapsed >= FADE_DURATION_MS) {
            return 0; // Sudah memudar total
        }
        // Linear fade: 255 → 0
        return (int) (255 * (1.0f - (float) fadeElapsed / FADE_DURATION_MS));
    }

    private void applyAlpha(int alpha) {
        mBoxPaint.setAlpha(alpha);
        mCirclePaint.setAlpha(alpha);
        mLinePaint.setAlpha(alpha);
        mTextPaint.setAlpha(alpha);
        mTextBgPaint.setAlpha((int)(alpha * 0.78f)); // bg sedikit lebih transparan
    }

    // =========================================================
    // LIFECYCLE: Start/Stop Choreographer loop
    // =========================================================

    @Override
    protected void onAttachedToWindow() {
        super.onAttachedToWindow();
        mChoreographerRunning = true;
        Choreographer.getInstance().postFrameCallback(mFrameCallback);
        Log.d(TAG, "Choreographer 60Hz loop started — Persistence of Vision aktif");
    }

    @Override
    protected void onDetachedFromWindow() {
        super.onDetachedFromWindow();
        mChoreographerRunning = false;
        Choreographer.getInstance().removeFrameCallback(mFrameCallback);
        Log.d(TAG, "Choreographer loop stopped");
    }

    // =========================================================
    // CONSTRUCTORS
    // =========================================================

    public OverlayView(Context context) {
        super(context);
        mBoxPaint    = createBoxPaint();
        mCirclePaint = createCirclePaint();
        mLinePaint   = createLinePaint();
        mTextPaint   = createTextPaint();
        mTextBgPaint = createTextBgPaint();
    }

    public OverlayView(Context context, AttributeSet attrs) {
        super(context, attrs);
        mBoxPaint    = createBoxPaint();
        mCirclePaint = createCirclePaint();
        mLinePaint   = createLinePaint();
        mTextPaint   = createTextPaint();
        mTextBgPaint = createTextBgPaint();
    }

    // =========================================================
    // FACTORY METHODS PAINT
    // =========================================================

    private Paint createBoxPaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.argb(255, 30, 144, 255));
        p.setStyle(Paint.Style.STROKE);
        p.setStrokeWidth(4.0f);
        return p;
    }

    private Paint createCirclePaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.RED);
        p.setStyle(Paint.Style.FILL);
        return p;
    }

    private Paint createLinePaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.RED);
        p.setStyle(Paint.Style.STROKE);
        p.setStrokeWidth(3.0f);
        return p;
    }

    private Paint createTextPaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.WHITE);
        p.setTextSize(36.0f);
        p.setTextAlign(Paint.Align.LEFT);
        return p;
    }

    private Paint createTextBgPaint() {
        Paint p = new Paint(Paint.ANTI_ALIAS_FLAG);
        p.setColor(Color.argb(200, 0, 0, 0));
        p.setStyle(Paint.Style.FILL);
        return p;
    }

    private void drawLabelWithBackground(Canvas canvas, String text, float x, float y, int alpha) {
        mTextPaint.getTextBounds(text, 0, text.length(), mTextBounds);
        float textW = mTextBounds.width();
        float textH = mTextBounds.height();
        float pad   = 6f;
        canvas.drawRect(x - pad, y - textH - pad, x + textW + pad, y + pad, mTextBgPaint);
        canvas.drawText(text, x, y, mTextPaint);
    }
}

