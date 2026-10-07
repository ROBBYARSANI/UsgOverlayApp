package com.example.usgandroidapp;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * UNetDetector.java
 * ==================
 * Menangani pipeline segmentasi menggunakan MobileNet-UNet (usg_vein_unet.tflite).
 *
 * Perbedaan fundamental dengan YOLOv8-seg:
 * - YOLOv8-seg: menghasilkan bounding box + mask coefficients → perlu matrix multiply dengan prototypes
 * - MobileNet-UNet: langsung menghasilkan pixel-wise probability map berukuran sama dengan input
 *
 * SPESIFIKASI MODEL:
 * - Input  : [1, 640, 640, 3] Float32, RGB, NHWC, normalized 0.0–1.0
 * - Output : [1, 640, 640, 1] Float32 — probability per piksel (0.0 = bukan vena, 1.0 = vena)
 *
 * CARA KERJA:
 * 1. Crop & resize citra USG ke 640×640 (letterbox)
 * 2. Normalisasi & masukkan ke buffer input
 * 3. Jalankan inferensi → dapatkan mask 640×640
 * 4. Threshold mask (nilai > MASK_THRESHOLD = vena)
 * 5. Hitung bounding box dari area aktif mask
 * 6. Scale koordinat kembali ke ruang layar asli
 */
public class UNetDetector {

    private static final String TAG = "UNetDetector";

    // ---------------------------------------------------------------
    // KONSTANTA MODEL
    // ---------------------------------------------------------------
    private static final String MODEL_PATH       = "usg_vein_unet.tflite";
    private static final int    MODEL_INPUT_SIZE = 640;
    private static final int    NUM_CHANNELS     = 3;

    // ---------------------------------------------------------------
    // THRESHOLD
    // ---------------------------------------------------------------
    /** Piksel dengan probabilitas di atas nilai ini dianggap vena */
    private static final float MASK_THRESHOLD   = 0.5f;
    /** Abaikan deteksi jika area aktif terlalu kecil (noise) */
    private static final int   MIN_MASK_PIXELS  = 200;

    // ---------------------------------------------------------------
    // BUFFER REUSABLE
    // ---------------------------------------------------------------
    /** Input NHWC: [1, 640, 640, 3] × float32 = 4,915,200 bytes */
    private final ByteBuffer  mInputBuffer;
    private final FloatBuffer mInputFloatBuffer;

    /** Output: [1, 640, 640, 1] — probability map */
    private final float[][][][] mOutputMask;

    // Pre-allocated arrays (menghindari GC per frame)
    private final int[]   mPixels;
    private final float[] mRgb; // Interleaved RGB untuk NHWC

    // ---------------------------------------------------------------
    // INTERPRETER & DELEGATE
    // ---------------------------------------------------------------
    private Interpreter mInterpreter;
    private GpuDelegate mGpuDelegate;

    // =========================================================
    // DATA CLASS HASIL DETEKSI UNET
    // =========================================================

    /**
     * Hasil deteksi UNet.
     * Berisi pixel mask langsung (sudah dalam koordinat overlay/screen),
     * ditambah bounding box yang dihitung dari kontur mask.
     */
    // =========================================================
    // DATA CLASS HASIL DETEKSI UNET
    // =========================================================

    public static class Blob {
        public float left, top, right, bottom, cx, cy, depthMm, confidence;
        public Blob(float left, float top, float right, float bottom,
                    float cx, float cy, float depthMm, float confidence) {
            this.left = left;
            this.top = top;
            this.right = right;
            this.bottom = bottom;
            this.cx = cx;
            this.cy = cy;
            this.depthMm = depthMm;
            this.confidence = confidence;
        }
    }

    /**
     * Hasil deteksi UNet.
     * Berisi pixel mask global (untuk seluruh vena),
     * dan daftar Blob (komponen terhubung) untuk bounding box masing-masing vena.
     */
    public static class UNetResult {
        public final Bitmap maskBitmap;
        public final java.util.List<Blob> blobs;

        public final float scaleX, scaleY, padLeft, padTop;
        public final int roiLeft, roiTop, roiWidth, roiHeight;

        public UNetResult(Bitmap maskBitmap, java.util.List<Blob> blobs,
                          float scaleX, float scaleY, float padLeft, float padTop,
                          int roiLeft, int roiTop, int roiWidth, int roiHeight) {
            this.maskBitmap = maskBitmap;
            this.blobs      = blobs;
            this.scaleX     = scaleX;
            this.scaleY     = scaleY;
            this.padLeft    = padLeft;
            this.padTop     = padTop;
            this.roiLeft    = roiLeft;
            this.roiTop     = roiTop;
            this.roiWidth   = roiWidth;
            this.roiHeight  = roiHeight;
        }
    }

    // =========================================================
    // KONSTRUKTOR
    // =========================================================

    private final float[] mColorLut = new float[256];
    private final int[] mMaskPixels = new int[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE];
    private final int[] qx = new int[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE];
    private final int[] qy = new int[MODEL_INPUT_SIZE * MODEL_INPUT_SIZE];
    private final boolean[][] visited = new boolean[MODEL_INPUT_SIZE][MODEL_INPUT_SIZE];
    private final boolean[][] binaryMask = new boolean[MODEL_INPUT_SIZE][MODEL_INPUT_SIZE];

    public UNetDetector(Context context) throws IOException {
        // Input NHWC: 1 * 640 * 640 * 3 * 4 bytes
        int bufferSize    = 1 * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * NUM_CHANNELS * 4;
        mInputBuffer      = ByteBuffer.allocateDirect(bufferSize);
        mInputBuffer.order(ByteOrder.nativeOrder());
        mInputFloatBuffer = mInputBuffer.asFloatBuffer();

        // Output: [1][640][640][1]
        mOutputMask = new float[1][MODEL_INPUT_SIZE][MODEL_INPUT_SIZE][1];

        // Pre-alokasi array piksel dan RGB
        int totalPixels = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
        mPixels = new int[totalPixels];
        mRgb    = new float[totalPixels * NUM_CHANNELS]; // NHWC: interleaved R,G,B per piksel

        MappedByteBuffer modelBuffer = loadModelFile(context, MODEL_PATH);
        Interpreter.Options options  = buildInterpreterOptions();
        mInterpreter = new Interpreter(modelBuffer, options);

        Log.d(TAG, "UNetDetector initialized. Model: " + MODEL_PATH);
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        android.content.res.AssetFileDescriptor fd = context.getAssets().openFd(modelPath);
        FileInputStream  is  = new FileInputStream(fd.getFileDescriptor());
        FileChannel      fc  = is.getChannel();
        return fc.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
    }

    private Interpreter.Options buildInterpreterOptions() {
        Interpreter.Options options = new Interpreter.Options();
        CompatibilityList compatList = new CompatibilityList();
        if (compatList.isDelegateSupportedOnThisDevice()) {
            Log.d(TAG, "GPU delegate supported");
            mGpuDelegate = new GpuDelegate(compatList.getBestOptionsForThisDevice());
            options.addDelegate(mGpuDelegate);
        } else {
            Log.d(TAG, "CPU mode, threads=" + Runtime.getRuntime().availableProcessors());
            options.setNumThreads(Runtime.getRuntime().availableProcessors());
        }
        // Gunakan NNAPI jika tersedia, lebih ngebut di device lama
        options.setUseNNAPI(true);
        return options;
    }

    // =========================================================
    // INFERENSI
    // =========================================================

    public UNetResult detect(Bitmap screenBitmap, int roiLeft, int roiTop,
                             int roiWidth, int roiHeight) {
        // STEP 1: Crop ke ROI
        Bitmap roiBitmap = Bitmap.createBitmap(screenBitmap, roiLeft, roiTop, roiWidth, roiHeight);

        // STEP 2: Letterbox resize ke 640×640
        LetterboxResult lb = letterboxResize(roiBitmap, MODEL_INPUT_SIZE);
        roiBitmap.recycle();

        // STEP 3: Konversi ke ByteBuffer NHWC (Optimized dengan LUT)
        bitmapToByteBufferNHWC(lb.bitmap);
        lb.bitmap.recycle();

        // STEP 4: Inferensi
        mInterpreter.run(mInputBuffer, mOutputMask);

        // STEP 5: Ambil mask, threshold, hitung bounding box (Optimized BFS & Bitmap Draw)
        return parseMaskAndBuildResult(mOutputMask[0], lb, roiLeft, roiTop, roiWidth, roiHeight);
    }

    public UNetResult detect(Bitmap screenBitmap) {
        return detect(screenBitmap, 0, 0, screenBitmap.getWidth(), screenBitmap.getHeight());
    }

    // =========================================================
    // LETTERBOX
    // =========================================================

    private static class LetterboxResult {
        Bitmap bitmap;
        float scaleX, scaleY, padLeft, padTop;
    }

    private LetterboxResult letterboxResize(Bitmap src, int targetSize) {
        int srcW  = src.getWidth();
        int srcH  = src.getHeight();
        float scale = Math.min((float) targetSize / srcW, (float) targetSize / srcH);
        int scaledW = Math.round(srcW * scale);
        int scaledH = Math.round(srcH * scale);
        int padLeft = (targetSize - scaledW) / 2;
        int padTop  = (targetSize - scaledH) / 2;

        Bitmap result = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888);
        android.graphics.Canvas canvas = new android.graphics.Canvas(result);
        canvas.drawColor(android.graphics.Color.BLACK);
        Bitmap scaled = Bitmap.createScaledBitmap(src, scaledW, scaledH, true);
        canvas.drawBitmap(scaled, padLeft, padTop, null);
        scaled.recycle();

        LetterboxResult lb = new LetterboxResult();
        lb.bitmap  = result;
        lb.scaleX  = scale;
        lb.scaleY  = scale;
        lb.padLeft = padLeft;
        lb.padTop  = padTop;
        return lb;
    }

    // =========================================================
    // PREPROCESSING: Bitmap → ByteBuffer NHWC Float32
    // =========================================================

    private void bitmapToByteBufferNHWC(Bitmap bitmap) {
        int w = bitmap.getWidth();
        int h = bitmap.getHeight();
        bitmap.getPixels(mPixels, 0, w, 0, 0, w, h);

        int numPixels = w * h;

        // Contrast Stretch (Auto-Level) — meningkatkan kontras citra USG gelap
        int minPix = 255, maxPix = 0;
        for (int i = 0; i < numPixels; i++) {
            int p = mPixels[i] & 0xFF;
            if (p < minPix) minPix = p;
            if (p > maxPix) maxPix = p;
        }
        float range = maxPix - minPix;
        if (range < 1.0f) range = 1.0f;
        float invRange = 1.0f / range;

        // Build Look-Up Table (LUT) to avoid 1.2 million float ops
        for (int p = 0; p < 256; p++) {
            mColorLut[p] = Math.max(0f, Math.min(1f, (p - minPix) * invRange));
        }

        // Format NHWC: untuk setiap piksel, tulis R, G, B berurutan
        int idx = 0;
        for (int i = 0; i < numPixels; i++) {
            int pixel = mPixels[i];
            mRgb[idx++] = mColorLut[(pixel >> 16) & 0xFF];
            mRgb[idx++] = mColorLut[(pixel >>  8) & 0xFF];
            mRgb[idx++] = mColorLut[ pixel        & 0xFF];
        }

        mInputFloatBuffer.rewind();
        mInputFloatBuffer.put(mRgb, 0, numPixels * NUM_CHANNELS);
    }

    // =========================================================
    // PARSE OUTPUT MASK & CONNECTED COMPONENTS
    // =========================================================

    private static final float SONOSTAR_DEPTH_MM = 20.0f;

    /**
     * Mengkonversi output mask model [640][640][1] menjadi UNetResult.
     */
    private UNetResult parseMaskAndBuildResult(float[][][] outputMask,
                                               LetterboxResult lb,
                                               int roiLeft, int roiTop,
                                               int roiWidth, int roiHeight) {
        // Reset buffers
        java.util.Arrays.fill(mMaskPixels, 0);
        for (int i = 0; i < MODEL_INPUT_SIZE; i++) {
            java.util.Arrays.fill(visited[i], false);
            java.util.Arrays.fill(binaryMask[i], false);
        }

        int validMinX = (int) lb.padLeft;
        int validMaxX = MODEL_INPUT_SIZE - (int) lb.padLeft;
        int validMinY = (int) lb.padTop;
        int validMaxY = MODEL_INPUT_SIZE - (int) lb.padTop;

        for (int y = validMinY; y < validMaxY; y++) {
            for (int x = validMinX; x < validMaxX; x++) {
                if (outputMask[y][x][0] > MASK_THRESHOLD) {
                    binaryMask[y][x] = true;
                }
            }
        }

        java.util.List<Blob> blobs = new java.util.ArrayList<>();
        int MASK_COLOR = 0x501E90FF; // DodgerBlue alpha 80

        for (int y = validMinY; y < validMaxY; y++) {
            for (int x = validMinX; x < validMaxX; x++) {
                if (binaryMask[y][x] && !visited[y][x]) {
                    // Temukan komponen terhubung dengan BFS
                    int head = 0, tail = 0;
                    qx[tail] = x;
                    qy[tail] = y;
                    tail++;
                    visited[y][x] = true;
                    mMaskPixels[y * MODEL_INPUT_SIZE + x] = MASK_COLOR;

                    int minMX = x, maxMX = x;
                    int minMY = y, maxMY = y;
                    int activePx = 0;
                    float confSum = 0f;

                    while (head < tail) {
                        int cx = qx[head];
                        int cy = qy[head];
                        head++;

                        activePx++;
                        confSum += outputMask[cy][cx][0];

                        if (cx < minMX) minMX = cx;
                        if (cx > maxMX) maxMX = cx;
                        if (cy < minMY) minMY = cy;
                        if (cy > maxMY) maxMY = cy;

                        // Cek tetangga 4-arah
                        if (cx > validMinX && binaryMask[cy][cx - 1] && !visited[cy][cx - 1]) {
                            visited[cy][cx - 1] = true;
                            mMaskPixels[cy * MODEL_INPUT_SIZE + (cx - 1)] = MASK_COLOR;
                            qx[tail] = cx - 1; qy[tail] = cy; tail++;
                        }
                        if (cx < validMaxX - 1 && binaryMask[cy][cx + 1] && !visited[cy][cx + 1]) {
                            visited[cy][cx + 1] = true;
                            mMaskPixels[cy * MODEL_INPUT_SIZE + (cx + 1)] = MASK_COLOR;
                            qx[tail] = cx + 1; qy[tail] = cy; tail++;
                        }
                        if (cy > validMinY && binaryMask[cy - 1][cx] && !visited[cy - 1][cx]) {
                            visited[cy - 1][cx] = true;
                            mMaskPixels[(cy - 1) * MODEL_INPUT_SIZE + cx] = MASK_COLOR;
                            qx[tail] = cx; qy[tail] = cy - 1; tail++;
                        }
                        if (cy < validMaxY - 1 && binaryMask[cy + 1][cx] && !visited[cy + 1][cx]) {
                            visited[cy + 1][cx] = true;
                            mMaskPixels[(cy + 1) * MODEL_INPUT_SIZE + cx] = MASK_COLOR;
                            qx[tail] = cx; qy[tail] = cy + 1; tail++;
                        }
                    }

                    // Hanya daftarkan komponen yang cukup besar
                    if (activePx >= MIN_MASK_PIXELS) {
                        float avgConf = confSum / activePx;

                        // Reverse-map bbox dari model space → ROI space
                        float leftRoi   = (minMX - lb.padLeft) / lb.scaleX;
                        float topRoi    = (minMY - lb.padTop)  / lb.scaleY;
                        float rightRoi  = (maxMX - lb.padLeft) / lb.scaleX;
                        float bottomRoi = (maxMY - lb.padTop)  / lb.scaleY;

                        // Clamp ke batas ROI
                        leftRoi   = Math.max(0, Math.min(leftRoi,   roiWidth));
                        topRoi    = Math.max(0, Math.min(topRoi,    roiHeight));
                        rightRoi  = Math.max(0, Math.min(rightRoi,  roiWidth));
                        bottomRoi = Math.max(0, Math.min(bottomRoi, roiHeight));

                        // ROI space → screen space
                        float leftScreen   = leftRoi   + roiLeft;
                        float topScreen    = topRoi    + roiTop;
                        float rightScreen  = rightRoi  + roiLeft;
                        float bottomScreen = bottomRoi + roiTop;
                        float cxScreen     = (leftScreen + rightScreen) / 2f;
                        float cyScreen     = (topScreen  + bottomScreen) / 2f;

                        // Hitung kedalaman
                        float cyRelative = cyScreen - roiTop;
                        float depthMm    = Math.max(0, Math.min((cyRelative / roiHeight) * SONOSTAR_DEPTH_MM, SONOSTAR_DEPTH_MM));

                        blobs.add(new Blob(leftScreen, topScreen, rightScreen, bottomScreen,
                                cxScreen, cyScreen, depthMm, avgConf));
                    } else {
                        // Hapus dari mask pixels agar tidak dirender
                        for (int i = 0; i < tail; i++) {
                            mMaskPixels[qy[i] * MODEL_INPUT_SIZE + qx[i]] = 0;
                        }
                    }
                }
            }
        }

        if (blobs.isEmpty()) return null;

        // Convert the final ARGB int array to a Bitmap (sangat cepat ~1ms)
        Bitmap maskBitmap = Bitmap.createBitmap(mMaskPixels, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, Bitmap.Config.ARGB_8888);

        return new UNetResult(
                maskBitmap, blobs,
                lb.scaleX, lb.scaleY, lb.padLeft, lb.padTop,
                roiLeft, roiTop, roiWidth, roiHeight
        );
    }

    // =========================================================
    // RESOURCE MANAGEMENT
    // =========================================================

    public void close() {
        if (mInterpreter != null) { mInterpreter.close(); mInterpreter = null; }
        if (mGpuDelegate  != null) { mGpuDelegate.close();  mGpuDelegate  = null; }
        Log.d(TAG, "UNetDetector closed");
    }
}
