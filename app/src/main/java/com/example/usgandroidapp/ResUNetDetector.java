package com.example.usgandroidapp;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.util.Log;

import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.CompatibilityList;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * ResUNetDetector.java — Optimized Edition
 * ==========================================
 * Pipeline inferensi untuk Mobile Residual U-Net dengan 3 optimasi utama:
 *
 * [1] GPU DELEGATE (FP16):
 *     Inferensi dijalankan di GPU menggunakan GpuDelegate dengan setPrecisionLossAllowed(true).
 *     Ini memungkinkan model FP16 berjalan native di GPU tanpa konversi float32 penuh.
 *     Pada chip dengan Adreno/Mali GPU, ini 3-5x lebih cepat dari CPU.
 *     Fallback otomatis ke CPU 4-thread jika GPU tidak mendukung.
 *
 * [2] ZERO-GC BUFFERS (Pre-allocated):
 *     mInputBuffer, mOutput, mPixels, mMaskPixels semuanya dialokasikan SEKALI
 *     di konstruktor. Tidak ada alokasi heap baru di dalam loop inferensi.
 *     Ini mencegah Garbage Collector (GC) dari menyebabkan frame drop secara acak.
 *
 * [3] MASK BITMAP CREATION (Efisiensi):
 *     Array int[] maskPixels direuse per-frame. Bitmap baru tetap dibuat per-frame
 *     (diperlukan karena dikirim ke OverlayView), tapi pixel array tidak dialokasikan ulang.
 *
 * SPESIFIKASI MODEL (Mobile_ResUNet_FP16.tflite):
 * - Input  : [1, 256, 256, 3]  float32, NHWC, normalized 0.0-1.0
 * - Output : [1, 256, 256, 1]  float32, Sigmoid (0.0-1.0)
 */
public class ResUNetDetector {

    private static final String TAG = "ResUNetDetector";

    // ---------------------------------------------------------------
    // KONSTANTA MODEL
    // ---------------------------------------------------------------
    private static final String MODEL_PATH       = "Mobile_ResUNet_FP16.tflite";
    private static final int    INPUT_SIZE        = 256;
    private static final int    NUM_CHANNELS      = 3;
    private static final float  MASK_THRESHOLD    = 0.5f;
    private static final float  MIN_BOX_SIZE_ROI  = 10f;
    private static final float  SONOSTAR_DEPTH_MM = 20.0f;

    // Warna mask vena: DodgerBlue, alpha 120 (semi-transparan)
    private static final int VEIN_COLOR = Color.argb(120, 30, 144, 255);

    // ---------------------------------------------------------------
    // BUFFER PRE-ALLOCATED — tidak ada alokasi baru per-frame (Zero-GC)
    // ---------------------------------------------------------------
    /** Input NHWC: [1,256,256,3] × float32 = 786,432 bytes. Dialokasikan sekali. */
    private final ByteBuffer  mInputBuffer;
    /** Output TFLite: [1][256][256][1]. Dialokasikan sekali, diisi ulang tiap inference. */
    private final float[][][][] mOutput;
    /** Buffer piksel integer: 256×256 = 65,536 elemen. Dialokasikan sekali. */
    private final int[] mPixels;
    /**
     * Buffer piksel mask berwarna: 256×256 = 65,536 elemen.
     * Diisi ulang tiap frame tanpa alokasi baru. Mencegah tekanan pada GC.
     */
    private final int[] mMaskPixels;

    // LUT normalisasi: O(1) lookup, tidak ada operasi pembagian per piksel
    private static final float[] INV255_LUT = new float[256];
    static {
        for (int i = 0; i < 256; i++) INV255_LUT[i] = i / 255.0f;
    }

    // ---------------------------------------------------------------
    // INTERPRETER & GPU DELEGATE
    // ---------------------------------------------------------------
    private Interpreter mInterpreter;
    /**
     * GpuDelegate — harus disimpan sebagai field karena TFLite memegang referensi
     * ke delegate secara internal. Melepasnya sebelum interpreter di-close akan crash.
     */
    private GpuDelegate mGpuDelegate = null;

    // =========================================================
    // DATA CLASS HASIL DETEKSI
    // =========================================================

    /**
     * Hasil inferensi ResUNet untuk satu frame.
     * maskBitmap: bitmap 256×256 berwarna biru menampilkan area vena secara piksel-akurat.
     */
    public static class Detection {
        public float left, top, right, bottom; // bbox di screen space
        public float cx, cy;                   // centroid di screen space
        public float confidence;               // rata-rata probabilitas area vena
        public float depthMm;                  // estimasi kedalaman dalam mm
        public Bitmap maskBitmap;              // mask segmentasi untuk overlay
        public int roiLeft, roiTop, roiWidth, roiHeight; // info ROI
    }

    // =========================================================
    // KONSTRUKTOR
    // =========================================================

    public ResUNetDetector(Context context) throws IOException {
        // ---- Pre-alokasi semua buffer (hanya terjadi SEKALI) ----
        int inputBytes = INPUT_SIZE * INPUT_SIZE * NUM_CHANNELS * Float.BYTES; // 786,432 bytes
        mInputBuffer = ByteBuffer.allocateDirect(inputBytes);
        mInputBuffer.order(ByteOrder.nativeOrder());

        mOutput    = new float[1][INPUT_SIZE][INPUT_SIZE][1]; // 256KB output
        mPixels    = new int[INPUT_SIZE * INPUT_SIZE];        // 256KB pixel buffer
        mMaskPixels= new int[INPUT_SIZE * INPUT_SIZE];        // 256KB mask buffer (REUSED)

        // ---- Load model & buat interpreter dengan GPU delegate ----
        MappedByteBuffer modelBuffer = loadModelFile(context);
        Interpreter.Options options  = buildInterpreterOptions();
        mInterpreter = new Interpreter(modelBuffer, options);

        Log.d(TAG, "ResUNetDetector initialized [" + MODEL_PATH + "]"
                + " GPU=" + (mGpuDelegate != null));
    }

    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        android.content.res.AssetFileDescriptor fd = context.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fd.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY,
                fd.getStartOffset(), fd.getDeclaredLength());
    }

    /**
     * [OPTIMASI 1] GPU Hardware Acceleration.
     *
     * Menggunakan CompatibilityList untuk mendeteksi kemampuan GPU secara runtime,
     * lalu membuat GpuDelegate dengan konfigurasi terbaik untuk perangkat ini.
     * setPrecisionLossAllowed(true) = izinkan FP16 native di GPU (krusial untuk model FP16).
     *
     * Hierarchy fallback:
     *   GPU (FP16) → CPU (4 threads)
     */
    private Interpreter.Options buildInterpreterOptions() {
        Interpreter.Options options = new Interpreter.Options();
        try {
            CompatibilityList compatList = new CompatibilityList();
            if (compatList.isDelegateSupportedOnThisDevice()) {
                // Dapatkan konfigurasi optimal GPU untuk perangkat ini
                GpuDelegate.Options gpuOptions = compatList.getBestOptionsForThisDevice();
                // Aktifkan FP16 native — model FP16 berjalan tanpa overhead konversi
                gpuOptions.setPrecisionLossAllowed(true);
                mGpuDelegate = new GpuDelegate(gpuOptions);
                options.addDelegate(mGpuDelegate);
                Log.d(TAG, "GPU Delegate aktif (FP16 precision loss allowed)");
            } else {
                // GPU tidak tersedia — fallback ke CPU multi-thread
                options.setNumThreads(4);
                Log.w(TAG, "GPU tidak tersedia, menggunakan CPU 4-thread");
            }
        } catch (Exception e) {
            // Tangkap error runtime GPU (driver crash, OOM, dll.)
            mGpuDelegate = null;
            options.setUseNNAPI(false);
            options.setNumThreads(4);
            Log.e(TAG, "GPU Delegate gagal, fallback CPU: " + e.getMessage());
        }
        return options;
    }

    // =========================================================
    // DETEKSI (PIPELINE UTAMA)
    // =========================================================

    /**
     * Jalankan full pipeline inferensi ResUNet untuk satu frame.
     * Method ini DIPANGGIL DI INFERENCE THREAD, bukan UI Thread.
     *
     * @param screenBitmap Frame layar lengkap.
     * @param roiLeft      Koordinat kiri ROI di layar.
     * @param roiTop       Koordinat atas ROI.
     * @param roiWidth     Lebar ROI.
     * @param roiHeight    Tinggi ROI.
     * @return Detection atau null jika tidak ada vena.
     */
    public Detection detect(Bitmap screenBitmap,
                            int roiLeft, int roiTop,
                            int roiWidth, int roiHeight) {
        // STEP 1: Crop ROI dari frame layar
        Bitmap roiBitmap = Bitmap.createBitmap(screenBitmap, roiLeft, roiTop, roiWidth, roiHeight);

        // STEP 2: Resize ke 256×256 (stretch sederhana sesuai cara training)
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(roiBitmap, INPUT_SIZE, INPUT_SIZE, true);
        roiBitmap.recycle();

        // STEP 3: Konversi Bitmap → ByteBuffer NHWC Float32 (menggunakan LUT, Zero-GC)
        bitmapToBufferNHWC(resizedBitmap);
        resizedBitmap.recycle();

        // STEP 4: Inferensi GPU/CPU
        // Wajib pass mOutput (float[1][256][256][1]) bukan mOutput[0] (float[256][256][1])
        // TFLite memerlukan batch dimension agar shape tensor cocok.
        mInterpreter.run(mInputBuffer, mOutput);

        // STEP 5: Post-process → Detection object
        return buildDetection(roiLeft, roiTop, roiWidth, roiHeight);
    }

    // =========================================================
    // PRE-PROCESSING: Bitmap → ByteBuffer NHWC Float32
    // =========================================================

    /**
     * Konversi Bitmap 256×256 ke format input model NHWC.
     * NHWC: untuk setiap piksel (h, w) → tulis R, G, B berurutan.
     *
     * Menggunakan INV255_LUT untuk menghindari operasi pembagian floating-point
     * sebanyak 196,608 kali per frame. Lookup table O(1) jauh lebih cepat.
     */
    private void bitmapToBufferNHWC(Bitmap bitmap) {
        // getPixels mengisi mPixels tanpa alokasi baru (reuse pre-allocated array)
        bitmap.getPixels(mPixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);

        mInputBuffer.rewind();
        final int numPixels = INPUT_SIZE * INPUT_SIZE;

        for (int i = 0; i < numPixels; i++) {
            final int pixel = mPixels[i];
            mInputBuffer.putFloat(INV255_LUT[(pixel >> 16) & 0xFF]); // R / 255.0
            mInputBuffer.putFloat(INV255_LUT[(pixel >>  8) & 0xFF]); // G / 255.0
            mInputBuffer.putFloat(INV255_LUT[ pixel        & 0xFF]); // B / 255.0
        }
    }

    // =========================================================
    // POST-PROCESSING: Output Mask → Detection Object
    // =========================================================

    /**
     * Konversi output tensor [1][256][256][1] menjadi:
     * 1. Bitmap mask berwarna (visualisasi vena biru semi-transparan).
     * 2. Bounding box minimum dari piksel vena.
     * 3. Koordinat centroid dan kedalaman.
     *
     * Menggunakan mMaskPixels yang sudah pre-allocated — tidak ada alokasi
     * int[] baru per frame, mencegah GC pause.
     */
    private Detection buildDetection(int roiLeft, int roiTop, int roiWidth, int roiHeight) {
        final float scaleX = (float) roiWidth  / INPUT_SIZE;
        final float scaleY = (float) roiHeight / INPUT_SIZE;
        final float[][][] outputSlice = mOutput[0]; // akses batch 0 sekali — float[256][256][1]

        int minH = Integer.MAX_VALUE, maxH = Integer.MIN_VALUE;
        int minW = Integer.MAX_VALUE, maxW = Integer.MIN_VALUE;
        float sumConf   = 0f;
        int   veinCount = 0;

        // Iterasi output mask — isi mMaskPixels (REUSED, bukan new int[])
        for (int h = 0; h < INPUT_SIZE; h++) {
            final float[][] row = outputSlice[h]; // cache row reference — float[256][1]
            for (int w = 0; w < INPUT_SIZE; w++) {
                final float prob = row[w][0]; // mOutput[0][h][w][0] — channel ke-0
                if (prob >= MASK_THRESHOLD) {
                    mMaskPixels[h * INPUT_SIZE + w] = VEIN_COLOR;
                    if (h < minH) minH = h;
                    if (h > maxH) maxH = h;
                    if (w < minW) minW = w;
                    if (w > maxW) maxW = w;
                    sumConf += prob;
                    veinCount++;
                } else {
                    mMaskPixels[h * INPUT_SIZE + w] = Color.TRANSPARENT;
                }
            }
        }

        if (veinCount == 0) return null;

        // Filter noise — bbox terlalu kecil diabaikan
        if ((maxW - minW) * scaleX < MIN_BOX_SIZE_ROI ||
            (maxH - minH) * scaleY < MIN_BOX_SIZE_ROI) return null;

        // Petakan koordinat: model space (256×256) → screen space
        final float leftScreen   = minW * scaleX + roiLeft;
        final float topScreen    = minH * scaleY + roiTop;
        final float rightScreen  = (maxW + 1) * scaleX + roiLeft;
        final float bottomScreen = (maxH + 1) * scaleY + roiTop;
        final float cxScreen     = (leftScreen + rightScreen)  / 2f;
        final float cyScreen     = (topScreen  + bottomScreen) / 2f;
        final float depthMm      = Math.max(0f, Math.min(
                ((cyScreen - roiTop) / roiHeight) * SONOSTAR_DEPTH_MM, SONOSTAR_DEPTH_MM));

        // Buat Bitmap mask dari mMaskPixels yang sudah terisi
        // (Bitmap baru per-frame diperlukan karena OverlayView memegang referensi ke bitmap sebelumnya)
        Bitmap maskBitmap = Bitmap.createBitmap(mMaskPixels, INPUT_SIZE, INPUT_SIZE,
                Bitmap.Config.ARGB_8888);

        Detection det = new Detection();
        det.left       = leftScreen;
        det.top        = topScreen;
        det.right      = rightScreen;
        det.bottom     = bottomScreen;
        det.cx         = cxScreen;
        det.cy         = cyScreen;
        det.confidence = sumConf / veinCount;
        det.depthMm    = depthMm;
        det.maskBitmap = maskBitmap;
        det.roiLeft    = roiLeft;
        det.roiTop     = roiTop;
        det.roiWidth   = roiWidth;
        det.roiHeight  = roiHeight;

        return det;
    }

    // =========================================================
    // RESOURCE MANAGEMENT
    // =========================================================

    /**
     * Wajib dipanggil saat service berhenti.
     * GpuDelegate HARUS di-close SETELAH interpreter di-close.
     * Urutan yang salah menyebabkan crash native.
     */
    public void close() {
        if (mInterpreter != null) { mInterpreter.close(); mInterpreter = null; }
        if (mGpuDelegate  != null) { mGpuDelegate.close();  mGpuDelegate  = null; }
        Log.d(TAG, "ResUNetDetector closed (GPU delegate released)");
    }
}
