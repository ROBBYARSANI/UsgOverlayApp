package com.example.usgandroidapp;

import android.graphics.Bitmap;
import android.graphics.PixelFormat;
import android.hardware.display.DisplayManager;
import android.hardware.display.VirtualDisplay;
import android.media.Image;
import android.media.ImageReader;
import android.media.projection.MediaProjection;
import android.os.Handler;
import android.os.SystemClock;

import java.nio.ByteBuffer;
import java.util.concurrent.atomic.AtomicBoolean;

/** Raw screen frames supplied by MediaProjection for the legacy/live path. */
public final class MediaProjectionFrameSource implements FrameSource {
    private final MediaProjection projection;
    private final int width;
    private final int height;
    private final int densityDpi;
    private final Handler callbackHandler;
    private final AtomicBoolean running = new AtomicBoolean(false);

    private ImageReader imageReader;
    private VirtualDisplay virtualDisplay;
    private Listener listener;
    private int frameIndex;

    public MediaProjectionFrameSource(MediaProjection projection, int width, int height,
                                      int densityDpi, Handler callbackHandler) {
        this.projection = projection;
        this.width = width;
        this.height = height;
        this.densityDpi = densityDpi;
        this.callbackHandler = callbackHandler;
    }

    @Override
    public void start(Listener listener) {
        if (!running.compareAndSet(false, true)) {
            throw new IllegalStateException("MediaProjectionFrameSource is already running");
        }
        this.listener = listener;
        frameIndex = 0;

        projection.registerCallback(new MediaProjection.Callback() {
            @Override
            public void onStop() {
                running.set(false);
                Listener current = MediaProjectionFrameSource.this.listener;
                if (current != null) current.onCompleted();
            }
        }, callbackHandler);

        imageReader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 2);
        imageReader.setOnImageAvailableListener(this::onImageAvailable, callbackHandler);
        virtualDisplay = projection.createVirtualDisplay(
                "VeinDisplay", width, height, densityDpi,
                DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
                imageReader.getSurface(), null, null);
    }

    private void onImageAvailable(ImageReader reader) {
        Image image = null;
        try {
            image = reader.acquireLatestImage();
            if (image == null || !running.get()) return;
            Bitmap bitmap = imageToBitmap(image, width, height);
            if (bitmap != null) {
                Listener current = listener;
                if (current != null) {
                    current.onFrame(new Frame(bitmap, frameIndex++, -1L,
                            SystemClock.elapsedRealtimeNanos()));
                } else if (!bitmap.isRecycled()) {
                    bitmap.recycle();
                }
            }
        } catch (Throwable error) {
            Listener current = listener;
            if (current != null) current.onError(error);
        } finally {
            if (image != null) image.close();
        }
    }

    static Bitmap imageToBitmap(Image image, int cropWidth, int cropHeight) {
        Image.Plane plane = image.getPlanes()[0];
        ByteBuffer buffer = plane.getBuffer();
        int pixelStride = plane.getPixelStride();
        int rowStride = plane.getRowStride();
        int rowPadding = rowStride - pixelStride * cropWidth;
        int paddedWidth = cropWidth + rowPadding / pixelStride;
        Bitmap padded = Bitmap.createBitmap(paddedWidth, cropHeight, Bitmap.Config.ARGB_8888);
        padded.copyPixelsFromBuffer(buffer);
        if (paddedWidth == cropWidth) return padded;
        Bitmap cropped = Bitmap.createBitmap(padded, 0, 0, cropWidth, cropHeight);
        padded.recycle();
        return cropped;
    }

    @Override
    public void stop() {
        if (!running.getAndSet(false) && imageReader == null && virtualDisplay == null) return;
        if (imageReader != null) {
            imageReader.setOnImageAvailableListener(null, null);
            imageReader.close();
            imageReader = null;
        }
        if (virtualDisplay != null) {
            virtualDisplay.release();
            virtualDisplay = null;
        }
        listener = null;
    }

    @Override
    public String getName() {
        return "MediaProjectionFrameSource";
    }
}
