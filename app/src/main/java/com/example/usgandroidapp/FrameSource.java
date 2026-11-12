package com.example.usgandroidapp;

import android.graphics.Bitmap;

/**
 * Source of raw, undecorated video frames.
 *
 * <p>The listener owns every delivered bitmap and must recycle it.  A source must never
 * receive a bitmap back from the presentation layer; this is the boundary that prevents
 * application overlays from becoming detector input.</p>
 */
public interface FrameSource extends AutoCloseable {

    final class Frame {
        public final Bitmap bitmap;
        public final int frameIndex;
        public final long mediaTimestampUs;
        public final long captureTimestampNs;

        public Frame(Bitmap bitmap, int frameIndex, long mediaTimestampUs,
                     long captureTimestampNs) {
            this.bitmap = bitmap;
            this.frameIndex = frameIndex;
            this.mediaTimestampUs = mediaTimestampUs;
            this.captureTimestampNs = captureTimestampNs;
        }
    }

    interface Listener {
        /** Called on the source worker thread. The listener takes ownership of frame.bitmap. */
        void onFrame(Frame frame);

        void onCompleted();

        void onError(Throwable error);
    }

    void start(Listener listener);

    void stop();

    String getName();

    @Override
    default void close() {
        stop();
    }
}
