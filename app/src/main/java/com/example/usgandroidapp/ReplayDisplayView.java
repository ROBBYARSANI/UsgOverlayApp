package com.example.usgandroidapp;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

import java.util.Locale;

/** Presentation-only canvas for clean replay. Its pixels are never read by a FrameSource. */
public final class ReplayDisplayView extends View {
    private final Paint bitmapPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final Paint boxPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint maskPaint = new Paint(Paint.FILTER_BITMAP_FLAG);
    private final Paint textPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint hudPaint = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF videoDst = new RectF();
    private final RectF maskDst = new RectF();

    private Bitmap frame;
    private YoloDetector.Detection detection;
    private boolean showHud;
    private int frameIndex;
    private long timestampUs;

    public ReplayDisplayView(Context context) {
        super(context);
        setBackgroundColor(Color.BLACK);
        boxPaint.setColor(Color.rgb(0, 240, 255));
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(4f);
        maskPaint.setAlpha(153);
        textPaint.setColor(Color.WHITE);
        textPaint.setTextSize(22f);
        hudPaint.setColor(Color.argb(210, 0, 0, 0));
    }

    /** Must be called on the UI thread. Takes ownership of frameCopy and detection.maskBitmap. */
    public void setPresentation(Bitmap frameCopy, YoloDetector.Detection detection,
                                boolean showHud, int frameIndex, long timestampUs) {
        releaseBitmaps();
        this.frame = frameCopy;
        this.detection = detection;
        this.showHud = showHud;
        this.frameIndex = frameIndex;
        this.timestampUs = timestampUs;
        invalidate();
    }

    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);
        Bitmap current = frame;
        if (current == null || current.isRecycled()) return;

        float scale = Math.min((float) getWidth() / current.getWidth(),
                (float) getHeight() / current.getHeight());
        float drawW = current.getWidth() * scale;
        float drawH = current.getHeight() * scale;
        float left = (getWidth() - drawW) * 0.5f;
        float top = (getHeight() - drawH) * 0.5f;
        videoDst.set(left, top, left + drawW, top + drawH);
        canvas.drawBitmap(current, null, videoDst, bitmapPaint);

        YoloDetector.Detection det = detection;
        if (det != null) {
            float l = left + det.left * scale;
            float t = top + det.top * scale;
            float r = left + det.right * scale;
            float b = top + det.bottom * scale;
            if (det.maskBitmap != null && !det.maskBitmap.isRecycled()) {
                maskDst.set(l, t, r, b);
                canvas.drawBitmap(det.maskBitmap, null, maskDst, maskPaint);
            }
            canvas.drawRect(l, t, r, b, boxPaint);
            canvas.drawText(String.format(Locale.US, "RAW %.3f", det.confidence),
                    l, Math.max(24f, t - 7f), textPaint);
        }

        if (showHud) {
            float h = 58f;
            canvas.drawRect(8f, getHeight() - h - 8f, 360f, getHeight() - 8f, hudPaint);
            canvas.drawText("REPLAY / RAW / NO MEDIAPROJECTION", 18f,
                    getHeight() - 39f, textPaint);
            canvas.drawText(String.format(Locale.US, "frame=%d  t=%.3fs", frameIndex,
                            timestampUs / 1_000_000.0),
                    18f, getHeight() - 15f, textPaint);
        }
    }

    private void releaseBitmaps() {
        if (frame != null && !frame.isRecycled()) frame.recycle();
        if (detection != null && detection.maskBitmap != null
                && !detection.maskBitmap.isRecycled()) {
            detection.maskBitmap.recycle();
        }
        frame = null;
        detection = null;
    }

    public void release() {
        releaseBitmaps();
    }
}
