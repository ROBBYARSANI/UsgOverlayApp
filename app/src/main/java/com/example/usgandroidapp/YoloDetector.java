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
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * YoloDetector.java
 * ==================
 * Menangani pipeline deteksi YOLOv8-Seg (Instance Segmentation)
 *
 * SPESIFIKASI MODEL (YoloV8_320.tflite - YOLOv8-seg 320x320):
 * - Input  : [1, 3, 320, 320] Float32, RGB, normalized 0.0-1.0, format NCHW
 * - Output 0: [1, 37, 2100] — 4 bbox + 1 conf + 32 mask coefficients per proposal
 * - Output 1: [1, 32, 80, 80] — 32 prototype mask maps
 *
 * KEUNGGULAN vs 640x640:
 * - Inferensi ~4x lebih cepat (estimasi 50-80ms vs 200-300ms di Tab A8)
 * - Blink duration < 80ms → mendekati batas persepsi mata manusia
 * - Kotak bbox mengikuti vena jauh lebih responsif dan "lengket"
 */
public class YoloDetector {

    private static final String TAG = "YoloDetector";

    // ---------------------------------------------------------------
    // KONSTANTA MODEL — sesuai YoloV8_320.tflite
    // ---------------------------------------------------------------
    private static final String MODEL_PATH       = "YoloV8_320.tflite";
    private static final int    MODEL_INPUT_SIZE = 320;   // ← 320 (bukan 640)
    private static final int    NUM_CHANNELS     = 3;
    private static final int    NUM_BOXES        = 2100;  // ← 2100 (bukan 8400), 4x lebih cepat parse
    /** Output 0 shape: [1, 37, 2100] */
    private static final int    NUM_COORDS_SEG   = 37;    // 4 bbox + 1 conf + 32 mask coeff
    private static final int    NUM_MASK_COEFF   = 32;
    /** Output 1 (prototype): 80×80 (bukan 160×160) */
    private static final int    PROTO_SIZE       = 80;    // ← 80 (bukan 160)

    // ---------------------------------------------------------------
    // THRESHOLD
    // ---------------------------------------------------------------
    // 0.25: cukup selektif, cukup rendah agar model sering berhasil detect
    private static final float CONFIDENCE_THRESHOLD = 0.25f;
    private static final float IOU_THRESHOLD        = 0.45f;
    private static final float MASK_THRESHOLD       = 0.45f;

    // ---------------------------------------------------------------
    // BUFFER REUSABLE
    // ---------------------------------------------------------------
    /** Input NCHW: [1, 3, 320, 320] × float32 = 1,228,800 bytes (4x lebih kecil!) */
    private final ByteBuffer  mInputBuffer;
    private final FloatBuffer mInputFloatBuffer;

    /** Output 0: deteksi + mask coefficients [1][37][2100] */
    private final float[][][] mOutputDet;
    /** Output 1: prototype masks [1][32][80][80] */
    private final float[][][][] mOutputProto;

    // Pre-allocated arrays (menghindari GC stutter per frame)
    private final int[]   mPixels;
    private final float[] mChannelR;
    private final float[] mChannelG;
    private final float[] mChannelB;

    // ---------------------------------------------------------------
    // INTERPRETER & DELEGATE
    // ---------------------------------------------------------------
    private Interpreter mInterpreter;
    private GpuDelegate mGpuDelegate;

    // =========================================================
    // DATA CLASS HASIL DETEKSI
    // =========================================================

    /**
     * Data class untuk satu hasil deteksi segmentasi.
     * Koordinat dalam satuan piksel di ruang layar asli.
     * maskBitmap adalah instance mask yang sudah di-scale ke koordinat overlay.
     */
    public static class Detection {
        public float left;
        public float top;
        public float right;
        public float bottom;
        public float cx;
        public float cy;
        public float confidence;
        public float depthMm;

        /**
         * Mask biner dalam ruang model prototype (160×160).
         * true = piksel tersebut adalah bagian dari vena.
         * Overlay akan me-remap nilai ini ke koordinat layar.
         */
        public boolean[][] maskProto;        // [PROTO_SIZE][PROTO_SIZE]
        /**
         * Bounding box di ruang ROI (bukan screen), digunakan untuk clipping mask.
         */
        public float leftRoi, topRoi, rightRoi, bottomRoi;
        /**
         * Mapping: posisi ROI → layar (untuk remap mask ke layar)
         */
        public int roiLeft, roiTop, roiWidth, roiHeight;

        public android.graphics.Bitmap maskBitmap;
        public float padLeft, padTop, scaleX, scaleY;

        public Detection(float left, float top, float right, float bottom,
                         float cx, float cy, float confidence, float depthMm,
                         boolean[][] maskProto,
                         float leftRoi, float topRoi, float rightRoi, float bottomRoi,
                         int roiLeft, int roiTop, int roiWidth, int roiHeight,
                         android.graphics.Bitmap maskBitmap,
                         float padLeft, float padTop, float scaleX, float scaleY) {
            this.left        = left;
            this.top         = top;
            this.right       = right;
            this.bottom      = bottom;
            this.cx          = cx;
            this.cy          = cy;
            this.confidence  = confidence;
            this.depthMm     = depthMm;
            this.maskProto   = maskProto;
            this.leftRoi     = leftRoi;
            this.topRoi      = topRoi;
            this.rightRoi    = rightRoi;
            this.bottomRoi   = bottomRoi;
            this.roiLeft     = roiLeft;
            this.roiTop      = roiTop;
            this.roiWidth    = roiWidth;
            this.roiHeight   = roiHeight;
            this.maskBitmap  = maskBitmap;
            this.padLeft     = padLeft;
            this.padTop      = padTop;
            this.scaleX      = scaleX;
            this.scaleY      = scaleY;
        }
    }

    // =========================================================
    // KONSTRUKTOR
    // =========================================================

    public YoloDetector(Context context) throws IOException {
        // Input NCHW: 1 * 3 * 640 * 640 * 4 bytes
        int bufferSize     = 1 * NUM_CHANNELS * MODEL_INPUT_SIZE * MODEL_INPUT_SIZE * 4;
        mInputBuffer       = ByteBuffer.allocateDirect(bufferSize);
        mInputBuffer.order(ByteOrder.nativeOrder());
        mInputFloatBuffer  = mInputBuffer.asFloatBuffer();

        // Output 0: [1][37][2100] — 4x lebih sedikit dari 640×640
        mOutputDet   = new float[1][NUM_COORDS_SEG][NUM_BOXES];
        // Output 1: [1][32][80][80] — prototype 80×80
        mOutputProto = new float[1][NUM_MASK_COEFF][PROTO_SIZE][PROTO_SIZE];

        // Pre-alokasi arrays piksel (320×320 = 102400, vs 409600 di 640×640)
        int totalPixels = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
        mPixels   = new int[totalPixels];
        mChannelR = new float[totalPixels];
        mChannelG = new float[totalPixels];
        mChannelB = new float[totalPixels];

        MappedByteBuffer modelBuffer = loadModelFile(context, MODEL_PATH);
        Interpreter.Options options  = buildInterpreterOptions();
        mInterpreter = new Interpreter(modelBuffer, options);

        Log.d(TAG, "YoloDetector 320x320 initialized. Model: " + MODEL_PATH
                + " | Estimasi inferensi ~4x lebih cepat dari 640x640");
    }

    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        android.content.res.AssetFileDescriptor fd = context.getAssets().openFd(modelPath);
        FileInputStream  inputStream  = new FileInputStream(fd.getFileDescriptor());
        FileChannel      fileChannel  = inputStream.getChannel();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fd.getStartOffset(), fd.getDeclaredLength());
    }

    private Interpreter.Options buildInterpreterOptions() {
        Interpreter.Options options = new Interpreter.Options();
        // NNAPI lebih konsisten & cepat di chipset Snapdragon 429 (Tab A8)
        // dibanding GPU delegate yang sering memiliki overhead inisialisasi tinggi.
        options.setUseNNAPI(true);
        options.setNumThreads(4);
        Log.d(TAG, "Inference: NNAPI enabled, 4 threads");
        return options;
    }

    // =========================================================
    // DETEKSI (PIPELINE UTAMA)
    // =========================================================

    public List<Detection> detect(Bitmap screenBitmap, int roiLeft, int roiTop,
                                  int roiWidth, int roiHeight) {
        // STEP 1: Crop ke ROI
        Bitmap roiBitmap = Bitmap.createBitmap(screenBitmap, roiLeft, roiTop, roiWidth, roiHeight);

        // STEP 2: Letterbox resize ke 640×640
        LetterboxResult letterbox = letterboxResize(roiBitmap, MODEL_INPUT_SIZE);
        roiBitmap.recycle();

        // STEP 3: Konversi Bitmap ke ByteBuffer NCHW Float32
        bitmapToByteBufferNCHW(letterbox.bitmap);
        letterbox.bitmap.recycle();

        // STEP 4: Inferensi — 2 output tensor
        mOutputDet[0]   = new float[NUM_COORDS_SEG][NUM_BOXES];
        mOutputProto[0] = new float[NUM_MASK_COEFF][PROTO_SIZE][PROTO_SIZE];

        Map<Integer, Object> outputMap = new HashMap<>();
        outputMap.put(0, mOutputDet);
        outputMap.put(1, mOutputProto);
        mInterpreter.runForMultipleInputsOutputs(new Object[]{mInputBuffer}, outputMap);

        // STEP 5: Parse raw boxes + mask coefficients
        List<float[]> rawBoxes = parseOutput(
                mOutputDet[0],
                letterbox.scaleX, letterbox.scaleY,
                letterbox.padLeft, letterbox.padTop,
                roiLeft, roiTop, roiWidth, roiHeight
        );

        // STEP 6: NMS
        List<float[]> nmsBoxes = applyNMS(rawBoxes);

        // STEP 7: Hitung mask per deteksi & bangun objek Detection
        return buildDetections(nmsBoxes, mOutputProto[0],
                letterbox.scaleX, letterbox.scaleY,
                letterbox.padLeft, letterbox.padTop,
                roiLeft, roiTop, roiWidth, roiHeight);
    }

    public List<Detection> detect(Bitmap screenBitmap) {
        return detect(screenBitmap, 0, 0, screenBitmap.getWidth(), screenBitmap.getHeight());
    }

    // =========================================================
    // LETTERBOX RESIZE
    // =========================================================

    private static class LetterboxResult {
        Bitmap bitmap;
        float scaleX;
        float scaleY;
        float padLeft;
        float padTop;
    }

    private LetterboxResult letterboxResize(Bitmap src, int targetSize) {
        int srcW = src.getWidth();
        int srcH = src.getHeight();

        float scale = Math.min((float) targetSize / srcW, (float) targetSize / srcH);
        int scaledW = Math.round(srcW * scale);
        int scaledH = Math.round(srcH * scale);

        int padLeft = (targetSize - scaledW) / 2;
        int padTop  = (targetSize - scaledH) / 2;

        // Canvas untuk gambar letterbox
        Bitmap result  = Bitmap.createBitmap(targetSize, targetSize, Bitmap.Config.ARGB_8888);
        android.graphics.Canvas canvas = new android.graphics.Canvas(result);
        canvas.drawColor(android.graphics.Color.BLACK); // padding hitam

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
    // PREPROCESSING: Bitmap → ByteBuffer NCHW Float32
    // =========================================================

    // LUT pra-buat sekali saat konstruktor (255 elemen, /255 per index)
    private static final float[] INV255_LUT = new float[256];
    static {
        for (int i = 0; i < 256; i++) INV255_LUT[i] = i / 255.0f;
    }

    private void bitmapToByteBufferNCHW(Bitmap bitmap) {
        int width  = bitmap.getWidth();
        int height = bitmap.getHeight();

        bitmap.getPixels(mPixels, 0, width, 0, 0, width, height);

        int numPixels = width * height;

        // Normalisasi standar: 0-255 → 0.0-1.0 menggunakan LUT (tidak pakai Contrast Stretch
        // karena 2× iterasi 409600 piksel + float division menyebabkan delay signifikan).
        for (int i = 0; i < numPixels; i++) {
            int pixel    = mPixels[i];
            mChannelR[i] = INV255_LUT[(pixel >> 16) & 0xFF];
            mChannelG[i] = INV255_LUT[(pixel >>  8) & 0xFF];
            mChannelB[i] = INV255_LUT[ pixel        & 0xFF];
        }

        // NCHW: tulis semua R, lalu semua G, lalu semua B
        mInputFloatBuffer.rewind();
        mInputFloatBuffer.put(mChannelR, 0, numPixels);
        mInputFloatBuffer.put(mChannelG, 0, numPixels);
        mInputFloatBuffer.put(mChannelB, 0, numPixels);
    }

    // =========================================================
    // STEP 5: PARSE OUTPUT
    // =========================================================

    /**
     * Parse output [37][8400].
     * Setiap box: [cx, cy, w, h, conf, coeff0..coeff31]
     * Kembalikan format: [left, top, right, bottom, cx, cy, conf, coeff0..31]
     * = total 7 + 32 = 39 elemen per entry.
     */
    private List<float[]> parseOutput(float[][] output,
                                      float scaleX, float scaleY,
                                      float padLeft, float padTop,
                                      int roiLeft, int roiTop,
                                      int roiWidth, int roiHeight) {
        List<float[]> boxes = new ArrayList<>();

        for (int boxIdx = 0; boxIdx < NUM_BOXES; boxIdx++) {
            float conf = output[4][boxIdx];
            if (conf <= CONFIDENCE_THRESHOLD) continue;

            // Model ekspor TFLite YOLOv8 biasanya output cx,cy,w,h dalam piksel model (0-640)
            float cxModel = output[0][boxIdx];
            float cyModel = output[1][boxIdx];
            float wModel  = output[2][boxIdx];
            float hModel  = output[3][boxIdx];

            // Jika range nilai kecil (<= 1), berarti sudah dinormalisasi → kalikan 640
            if (cxModel <= 1.0f && cyModel <= 1.0f && wModel <= 1.0f && hModel <= 1.0f) {
                cxModel *= MODEL_INPUT_SIZE;
                cyModel *= MODEL_INPUT_SIZE;
                wModel  *= MODEL_INPUT_SIZE;
                hModel  *= MODEL_INPUT_SIZE;
            }

            // Reverse letterbox → ROI space
            float cxRoi = (cxModel - padLeft) / scaleX;
            float cyRoi = (cyModel - padTop)  / scaleY;
            float wRoi  = wModel  / scaleX;
            float hRoi  = hModel  / scaleY;

            // Filter box hantu/noise kecil (yang lebarnya atau tingginya < 15 pixel)
            // Vena asli pasti lebih besar dari ukuran ini.
            if (wRoi < 15f || hRoi < 15f) {
                continue;
            }

            float leftRoi   = Math.max(0, Math.min(cxRoi - wRoi / 2f, roiWidth));
            float topRoi    = Math.max(0, Math.min(cyRoi - hRoi / 2f, roiHeight));
            float rightRoi  = Math.max(0, Math.min(cxRoi + wRoi / 2f, roiWidth));
            float bottomRoi = Math.max(0, Math.min(cyRoi + hRoi / 2f, roiHeight));

            // ROI → screen space
            float leftScreen   = leftRoi   + roiLeft;
            float topScreen    = topRoi    + roiTop;
            float rightScreen  = rightRoi  + roiLeft;
            float bottomScreen = bottomRoi + roiTop;
            float cxScreen     = cxRoi     + roiLeft;
            float cyScreen     = cyRoi     + roiTop;

            // Kemas: [left, top, right, bottom, cx, cy, conf, roiL, roiT, roiR, roiB, coeff0..31]
            float[] entry = new float[11 + NUM_MASK_COEFF]; // 11 + 32 = 43
            entry[0] = leftScreen;
            entry[1] = topScreen;
            entry[2] = rightScreen;
            entry[3] = bottomScreen;
            entry[4] = cxScreen;
            entry[5] = cyScreen;
            entry[6] = conf;
            entry[7] = leftRoi;
            entry[8] = topRoi;
            entry[9] = rightRoi;
            entry[10] = bottomRoi;
            for (int k = 0; k < NUM_MASK_COEFF; k++) {
                entry[11 + k] = output[5 + k][boxIdx];
            }
            boxes.add(entry);
        }
        return boxes;
    }

    // =========================================================
    // STEP 6: NMS
    // =========================================================

    private List<float[]> applyNMS(List<float[]> boxes) {
        if (boxes.isEmpty()) return boxes;
        boxes.sort((a, b) -> Float.compare(b[6], a[6]));

        List<float[]> kept       = new ArrayList<>();
        boolean[]     suppressed = new boolean[boxes.size()];

        for (int i = 0; i < boxes.size(); i++) {
            if (suppressed[i]) continue;
            float[] boxI = boxes.get(i);
            kept.add(boxI);
            for (int j = i + 1; j < boxes.size(); j++) {
                if (suppressed[j]) continue;
                if (calculateIoU(boxI, boxes.get(j)) >= IOU_THRESHOLD) {
                    suppressed[j] = true;
                }
            }
        }
        return kept;
    }

    private float calculateIoU(float[] a, float[] b) {
        float interLeft   = Math.max(a[0], b[0]);
        float interTop    = Math.max(a[1], b[1]);
        float interRight  = Math.min(a[2], b[2]);
        float interBottom = Math.min(a[3], b[3]);
        float interW = Math.max(0, interRight  - interLeft);
        float interH = Math.max(0, interBottom - interTop);
        float interArea = interW * interH;
        float areaA     = (a[2] - a[0]) * (a[3] - a[1]);
        float areaB     = (b[2] - b[0]) * (b[3] - b[1]);
        float unionArea = areaA + areaB - interArea;
        return unionArea <= 0 ? 0f : interArea / unionArea;
    }

    // =========================================================
    // STEP 7: HITUNG MASK & BANGUN OBJEK DETECTION
    // =========================================================

    private static final float SONOSTAR_DEPTH_MM = 20.0f;

    /**
     * Untuk setiap box yang lolos NMS:
     * 1. Kalikan mask_coefficients [32] × prototypes [32][160][160] → raw_mask [160][160]
     * 2. Sigmoid → nilai 0..1
     * 3. Threshold → boolean mask
     */
    private List<Detection> buildDetections(List<float[]> boxes,
                                            float[][][] proto,
                                            float scaleX, float scaleY,
                                            float padLeft, float padTop,
                                            int roiLeft, int roiTop,
                                            int roiWidth, int roiHeight) {
        List<Detection> detections = new ArrayList<>();

        for (float[] entry : boxes) {
            float leftScreen   = entry[0];
            float topScreen    = entry[1];
            float rightScreen  = entry[2];
            float bottomScreen = entry[3];
            float cxScreen     = entry[4];
            float cyScreen     = entry[5];
            float conf         = entry[6];
            float leftRoi      = entry[7];
            float topRoi2      = entry[8];
            float rightRoi     = entry[9];
            float bottomRoi    = entry[10];

            // Mask coefficients [32]
            float[] coeffs = new float[NUM_MASK_COEFF];
            for (int k = 0; k < NUM_MASK_COEFF; k++) {
                coeffs[k] = entry[11 + k];
            }

            // Kedalaman centroid
            float cyRelative = cyScreen - roiTop;
            float depthMm    = Math.max(0, Math.min((cyRelative / roiHeight) * SONOSTAR_DEPTH_MM, SONOSTAR_DEPTH_MM));

            // SKIP mask computation untuk kecepatan maksimal.
            // computeMaskBitmap() memanggil Math.exp() sebanyak ~25.600x per deteksi —
            // sangat berat di Snapdragon 429. Dengan tidak menghitungnya, siklus inferensi
            // jauh lebih cepat sehingga "blink" terjadi lebih singkat dan tidak terlihat.
            detections.add(new Detection(
                    leftScreen, topScreen, rightScreen, bottomScreen,
                    cxScreen, cyScreen, conf, depthMm,
                    null, // maskProto
                    leftRoi, topRoi2, rightRoi, bottomRoi,
                    roiLeft, roiTop, roiWidth, roiHeight,
                    null, padLeft, padTop, scaleX, scaleY // null maskBitmap = skip rendering
            ));
        }
        return detections;
    }

    /**
     * Kalkulasi instance mask untuk satu deteksi dan render langsung ke Bitmap 160x160.
     */
    private android.graphics.Bitmap computeMaskBitmap(float[] coeffs, float[][][] proto,
                                                      float scaleX, float scaleY,
                                                      float padLeft, float padTop,
                                                      float leftRoi, float topRoi, float rightRoi, float bottomRoi) {
        float protoScale = (float) PROTO_SIZE / MODEL_INPUT_SIZE;

        float leftModel   = leftRoi   * scaleX + padLeft;
        float topModel    = topRoi    * scaleY + padTop;
        float rightModel  = rightRoi  * scaleX + padLeft;
        float bottomModel = bottomRoi * scaleY + padTop;

        int clipL = Math.max(0, (int)(leftModel   * protoScale));
        int clipT = Math.max(0, (int)(topModel    * protoScale));
        int clipR = Math.min(PROTO_SIZE - 1, (int)(rightModel  * protoScale));
        int clipB = Math.min(PROTO_SIZE - 1, (int)(bottomModel * protoScale));

        int[] pixels = new int[PROTO_SIZE * PROTO_SIZE];
        int MASK_COLOR = 0x601E90FF; // DodgerBlue alpha ~100

        for (int py = clipT; py <= clipB; py++) {
            for (int px = clipL; px <= clipR; px++) {
                float val = 0f;
                for (int k = 0; k < NUM_MASK_COEFF; k++) {
                    val += coeffs[k] * proto[k][py][px];
                }
                float sig = 1f / (1f + (float) Math.exp(-val));
                if (sig > MASK_THRESHOLD) {
                    pixels[py * PROTO_SIZE + px] = MASK_COLOR;
                }
            }
        }
        return android.graphics.Bitmap.createBitmap(pixels, PROTO_SIZE, PROTO_SIZE, android.graphics.Bitmap.Config.ARGB_8888);
    }

    // =========================================================
    // RESOURCE MANAGEMENT
    // =========================================================

    public void close() {
        if (mInterpreter != null) { mInterpreter.close(); mInterpreter = null; }
        if (mGpuDelegate  != null) { mGpuDelegate.close();  mGpuDelegate  = null; }
        Log.d(TAG, "YoloDetector closed");
    }
}
