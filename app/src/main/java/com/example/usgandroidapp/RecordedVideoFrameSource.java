package com.example.usgandroidapp;

import android.graphics.Bitmap;
import android.media.MediaMetadataRetriever;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Deterministic Android-native MP4 source for replay audits.
 *
 * <p>Frames are selected by a fixed source-frame step, not wall-clock timing. This makes
 * separate model runs receive the same video timestamps and pixel buffers. No displayed
 * pixels or application windows are ever captured.</p>
 */
public final class RecordedVideoFrameSource implements FrameSource {
    private final String videoPath;
    private final int frameStep;
    private final AtomicBoolean running = new AtomicBoolean(false);

    private HandlerThread workerThread;
    private Handler worker;
    private MediaMetadataRetriever retriever;

    public RecordedVideoFrameSource(String videoPath, int frameStep) {
        if (videoPath == null || videoPath.trim().isEmpty()) {
            throw new IllegalArgumentException("videoPath is required");
        }
        this.videoPath = videoPath;
        this.frameStep = Math.max(1, frameStep);
    }

    @Override
    public void start(Listener listener) {
        if (!running.compareAndSet(false, true)) {
            throw new IllegalStateException("RecordedVideoFrameSource is already running");
        }
        workerThread = new HandlerThread("RecordedVideoFrameSource");
        workerThread.start();
        worker = new Handler(workerThread.getLooper());
        worker.post(() -> decode(listener));
    }

    private void decode(Listener listener) {
        try {
            retriever = new MediaMetadataRetriever();
            retriever.setDataSource(videoPath);

            int totalFrames = parseInt(
                    retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_VIDEO_FRAME_COUNT), -1);
            float fps = parseFloat(
                    retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_CAPTURE_FRAMERATE), 0f);
            long durationMs = parseLong(
                    retriever.extractMetadata(MediaMetadataRetriever.METADATA_KEY_DURATION), 0L);

            if (totalFrames <= 0 && fps > 0f && durationMs > 0L) {
                totalFrames = Math.max(1, Math.round(durationMs * fps / 1000f));
            }
            if (fps <= 0f && totalFrames > 1 && durationMs > 0L) {
                fps = totalFrames * 1000f / durationMs;
            }
            if (totalFrames <= 0 || fps <= 0f) {
                throw new IllegalStateException("MP4 metadata lacks frame count/fps");
            }

            for (int index = 0; running.get() && index < totalFrames; index += frameStep) {
                Bitmap bitmap;
                long timestampUs = Math.round(index * 1_000_000.0 / fps);
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
                    bitmap = retriever.getFrameAtIndex(index);
                } else {
                    bitmap = retriever.getFrameAtTime(timestampUs,
                            MediaMetadataRetriever.OPTION_CLOSEST);
                }
                if (bitmap == null) {
                    throw new IllegalStateException("Decoder returned null at frame " + index);
                }
                listener.onFrame(new Frame(bitmap, index, timestampUs,
                        SystemClock.elapsedRealtimeNanos()));
            }
            if (running.get()) listener.onCompleted();
        } catch (Throwable error) {
            if (running.get()) listener.onError(error);
        } finally {
            releaseRetriever();
            running.set(false);
        }
    }

    private static int parseInt(String value, int fallback) {
        try { return Integer.parseInt(value); } catch (Exception ignored) { return fallback; }
    }

    private static long parseLong(String value, long fallback) {
        try { return Long.parseLong(value); } catch (Exception ignored) { return fallback; }
    }

    private static float parseFloat(String value, float fallback) {
        try { return Float.parseFloat(value); } catch (Exception ignored) { return fallback; }
    }

    private void releaseRetriever() {
        if (retriever != null) {
            try { retriever.release(); } catch (Exception ignored) {}
            retriever = null;
        }
    }

    @Override
    public void stop() {
        running.set(false);
        releaseRetriever();
        if (workerThread != null) {
            workerThread.quitSafely();
            workerThread = null;
            worker = null;
        }
    }

    @Override
    public String getName() {
        return "RecordedVideoFrameSource(" + videoPath + ", step=" + frameStep + ")";
    }
}
