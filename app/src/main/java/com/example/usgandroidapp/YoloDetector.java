package com.example.usgandroidapp;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.os.SystemClock;
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
 * YoloDetector.java — YOLOv8-Seg Edition (YoloV8_320.tflite)
 * ============================================================
 * Pipeline inferensi YOLOv8 Instance Segmentation untuk deteksi vena USG.
 *
 * SPESIFIKASI MODEL (YoloV8_320.tflite — dikonfirmasi dari TFLite inspector):
 *   Input  : [1, 3, 320, 320] NCHW Float32, normalized 0.0–1.0
 *   Output0: [1, 37, 2100] → 4 bbox (cx,cy,w,h) + 1 conf + 32 mask-coeff per proposal
 *   Output1: [1, 32, 80, 80] → 32 prototype mask maps (80×80)
 *
 * PIPELINE:
 *   Frame → ROI Crop → Letterbox 320×320 → NCHW Buffer →
 *   TFLite Inference (GPU/CPU) → Parse + NMS → Mask decode → Detection
 *
 * OPTIMASI:
 *   [1] GPU Delegate (FP16 precision loss allowed) → fallback CPU 4-thread
 *   [2] Pre-allocated buffers (Zero-GC per frame)
 *   [3] LUT /255 normalisasi (tidak ada float division per piksel)
 *   [4] Letterbox (aspect-ratio preserving) agar tidak distorsi
 *   [5] NMS IoU-based untuk eliminasi duplikasi bbox
 */
public class YoloDetector {

    private static final String TAG = "YoloDetector";

    // ─────────────────────────────────────────────────────────────
    // KONSTANTA MODEL
    // ─────────────────────────────────────────────────────────────
    public static final String DEFAULT_MODEL_PATH = "YoloV8_320.tflite";
    private static final int    INPUT_SIZE        = 320;
    private static final int    NUM_CHANNELS      = 3;
    private static final int    NUM_PROPOSALS     = 2100;  // [1,37,2100]
    private static final int    NUM_ATTRS         = 37;    // 4+1+32
    private static final int    NUM_MASK_COEFF    = 32;
    private static final int    PROTO_H           = 80;
    private static final int    PROTO_W           = 80;

    // ─────────────────────────────────────────────────────────────
    // RUNTIME CONFIGURATION
    // ─────────────────────────────────────────────────────────────
    public static class RuntimeConfig {
        public String modelPath = DEFAULT_MODEL_PATH;
        public int numThreads = 4;
        /**
         * Set default useXnnpack = false:
         * Berdasarkan benchmark empiris pada Snapdragon 429 (quad-core A53),
         * penonaktifan XNNPACK memangkas latensi dari 372 ms ke 307 ms (+28.8% throughput)
         * dan mempercepat cold-start 4.7x (24 ms vs 114 ms) dengan IoU 1.000 bit-exact.
         */
        public boolean useXnnpack = false;
        public boolean enableGpu = false;
        public boolean enableNnapi = false;

        public RuntimeConfig() {}

        public RuntimeConfig(String modelPath, int numThreads, boolean useXnnpack) {
            this.modelPath = modelPath;
            this.numThreads = numThreads;
            this.useXnnpack = useXnnpack;
        }

        public RuntimeConfig(String modelPath, int numThreads, boolean useXnnpack, boolean enableNnapi) {
            this.modelPath = modelPath;
            this.numThreads = numThreads;
            this.useXnnpack = useXnnpack;
            this.enableNnapi = enableNnapi;
        }
    }

    private final String mModelPath;

    // ─────────────────────────────────────────────────────────────
    // THRESHOLD
    // ─────────────────────────────────────────────────────────────
    // Minimal proposal confidence yang masuk ke NMS (sebelum disaring Temporal Hysteresis 0.40/0.55)
    private static final float CONF_THRESHOLD = 0.35f;   // Ditingkatkan dari 0.25 untuk eliminasi proposal noise
    private static final float NMS_IOU_THRESH = 0.45f;   // NMS suppression threshold
    private static final float MASK_THRESHOLD = 0.45f;   // sigmoid threshold untuk piksel vena
    private static final float MIN_BOX_PX     = 12f;     // filter box noise sangat kecil

    // ─────────────────────────────────────────────────────────────
    // PRE-ALLOCATED BUFFERS (Zero-GC per frame)
    // ─────────────────────────────────────────────────────────────
    private final ByteBuffer  mInputBuffer;
    private final FloatBuffer mInputFloat;

    // Output 0: [1][37][2100]
    private final float[][][] mOutputDet;
    // Output 1: [1][32][80][80]
    private final float[][][][] mOutputProto;

    // Pixel & channel scratch arrays
    private final int[]   mPixels;
    private final float[] mChanR, mChanG, mChanB;

    // Mask pixel scratch (80×80 = 6400) — reused tiap mask decode
    private final int[]   mMaskPixels;

    // LUT normalisasi: O(1) /255
    private static final float[] INV255 = new float[256];
    static { for (int i = 0; i < 256; i++) INV255[i] = i / 255.0f; }

    // ─────────────────────────────────────────────────────────────
    // INTERPRETER & GPU DELEGATE
    // ─────────────────────────────────────────────────────────────
    private Interpreter mInterpreter;
    private GpuDelegate mGpuDelegate = null;
    private String mDelegateName = "Unknown";

    private static final float DEPTH_SCALE_MM = 20.0f;

    // ─────────────────────────────────────────────────────────────
    // PROFILING & TIMING DATA CLASS
    // ─────────────────────────────────────────────────────────────
    /**
     * Penampung durasi waktu (nanodetik) per tahap pemrosesan dalam satu frame.
     */
    public static class StageTimings {
        public long preprocessNs;
        public long inferenceNs;
        public long postprocessNs;
        public long maskDecodeNs;
    }

    public String getDelegateName() {
        return mDelegateName;
    }

    public String getModelName() {
        return mModelPath;
    }

    public int getInputSize() {
        return INPUT_SIZE;
    }

    // ─────────────────────────────────────────────────────────────
    // DATA CLASS
    // ─────────────────────────────────────────────────────────────
    /**
     * Hasil deteksi satu vena.
     * Semua koordinat dalam pixel space layar asli.
     */
    public static class Detection {
        public float left, top, right, bottom;
        public float cx, cy;
        public float confidence;
        public float depthMm;
        /** Mask instance yang sudah di-decode. null jika tidak ada mask. */
        public Bitmap maskBitmap;
        /** Informasi ROI untuk mapping mask ke layar di OverlayView. */
        public int roiLeft, roiTop, roiWidth, roiHeight;

        // --- TEMPORAL & DEBUG INSTRUMENTATION FIELDS ---
        public long captureTimestampNs = 0L;
        public long inferenceEndTimestampNs = 0L;
        public boolean isFresh = true;     // true if fresh from YOLO, false if held
        public double holdAgeMs = 0.0;     // time elapsed since last fresh YOLO detection
        public boolean isStale = false;    // marked stale if exceeding time validity threshold

        // RAW YOLO values (sebelum EMA)
        public float rawLeft, rawTop, rawRight, rawBottom;
        public float rawCx, rawCy;
        public float rawConfidence;

        // EMA smoothing metadata
        public float displacementPx = 0f;
        public float currentAlpha = 1.0f;

        public float width() { return Math.max(0f, right - left); }
        public float height() { return Math.max(0f, bottom - top); }
    }

    // ─────────────────────────────────────────────────────────────
    // KONSTRUKTOR
    // ─────────────────────────────────────────────────────────────

    public YoloDetector(Context context) throws IOException {
        this(context, new RuntimeConfig());
    }

    public YoloDetector(Context context, RuntimeConfig config) throws IOException {
        RuntimeConfig cfg = (config != null) ? config : new RuntimeConfig();
        this.mModelPath   = (cfg.modelPath != null) ? cfg.modelPath : DEFAULT_MODEL_PATH;

        // Input buffer: 1×3×320×320×4 bytes = 1,228,800 bytes
        int inputBytes = NUM_CHANNELS * INPUT_SIZE * INPUT_SIZE * Float.BYTES;
        mInputBuffer  = ByteBuffer.allocateDirect(inputBytes);
        mInputBuffer.order(ByteOrder.nativeOrder());
        mInputFloat   = mInputBuffer.asFloatBuffer();

        // Output buffers
        mOutputDet   = new float[1][NUM_ATTRS][NUM_PROPOSALS];
        mOutputProto = new float[1][NUM_MASK_COEFF][PROTO_H][PROTO_W];

        // Scratch arrays
        int totalPx   = INPUT_SIZE * INPUT_SIZE;
        mPixels       = new int[totalPx];
        mChanR        = new float[totalPx];
        mChanG        = new float[totalPx];
        mChanB        = new float[totalPx];
        mMaskPixels   = new int[PROTO_H * PROTO_W];

        MappedByteBuffer model   = loadModel(context);
        Interpreter.Options opts = buildOptions(cfg);
        mInterpreter = new Interpreter(model, opts);

        Log.d(TAG, "YoloDetector ready [" + mModelPath + "] " + mDelegateName);
    }

    private MappedByteBuffer loadModel(Context ctx) throws IOException {
        android.content.res.AssetFileDescriptor fd = ctx.getAssets().openFd(mModelPath);
        FileInputStream fis = new FileInputStream(fd.getFileDescriptor());
        return fis.getChannel().map(FileChannel.MapMode.READ_ONLY,
                fd.getStartOffset(), fd.getDeclaredLength());
    }

    private Interpreter.Options buildOptions(RuntimeConfig cfg) {
        Interpreter.Options opts = new Interpreter.Options();

        if (cfg.enableGpu) {
            try {
                CompatibilityList compat = new CompatibilityList();
                if (compat.isDelegateSupportedOnThisDevice()) {
                    GpuDelegate.Options gpuOpts = compat.getBestOptionsForThisDevice();
                    gpuOpts.setPrecisionLossAllowed(true);
                    mGpuDelegate = new GpuDelegate(gpuOpts);
                    opts.addDelegate(mGpuDelegate);
                    mDelegateName = "GPU Delegate (FP16)";
                    Log.d(TAG, "GPU delegate aktif (FP16)");
                    return opts;
                } else {
                    Log.w(TAG, "GPU tidak kompatibel untuk model ini");
                }
            } catch (Exception e) {
                mGpuDelegate = null;
                Log.e(TAG, "GPU delegate error: " + e.getMessage());
            }
        }

        if (cfg.enableNnapi) {
            try {
                opts.setUseNNAPI(true);
                mDelegateName = "NNAPI Delegate";
                Log.d(TAG, "NNAPI delegate diaktifkan");
                return opts;
            } catch (Exception e) {
                Log.e(TAG, "NNAPI failed: " + e.getMessage());
            }
        }

        // CPU Execution (XNNPACK / Reference)
        opts.setNumThreads(cfg.numThreads);
        opts.setUseXNNPACK(cfg.useXnnpack);
        mDelegateName = "CPU (" + cfg.numThreads + "T, XNNPACK=" + cfg.useXnnpack + ")";
        Log.d(TAG, "Configured: " + mDelegateName);
        return opts;
    }

    // ─────────────────────────────────────────────────────────────
    // PUBLIC API: detect
    // ─────────────────────────────────────────────────────────────

    /**
     * Jalankan inferensi YOLOv8-Seg pada satu frame dengan pengumpulan timing per tahap
     * dan pencatatan capture timestamp untuk end-to-end prediction age profiling.
     * Dipanggil dari Inference Thread, BUKAN UI thread.
     *
     * @param timings Penampung durasi waktu (nanodetik) per tahap. Boleh null jika tidak profiling.
     * @param captureTimestampNs Waktu frame di-capture oleh kamera (SystemClock.elapsedRealtimeNanos()).
     * @return List<Detection> yang lolos NMS. Kosong jika tidak ada vena.
     */
    public List<Detection> detect(Bitmap screenBitmap,
                                  int roiLeft, int roiTop,
                                  int roiWidth, int roiHeight,
                                  StageTimings timings,
                                  long captureTimestampNs) {
        long t0 = SystemClock.elapsedRealtimeNanos();

        // 1. Crop ROI
        Bitmap roiBmp = Bitmap.createBitmap(screenBitmap, roiLeft, roiTop, roiWidth, roiHeight);

        // 2. Letterbox → 320×320 (aspect-ratio preserving, pad hitam)
        LetterboxResult lb = letterbox(roiBmp, INPUT_SIZE);
        roiBmp.recycle();

        // 3. Bitmap → ByteBuffer NCHW float32
        encodeNCHW(lb.bitmap);
        lb.bitmap.recycle();

        long t1 = SystemClock.elapsedRealtimeNanos();
        if (timings != null) timings.preprocessNs = t1 - t0;

        // 4. Inferensi — dual output
        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(0, mOutputDet);
        outputs.put(1, mOutputProto);
        mInterpreter.runForMultipleInputsOutputs(new Object[]{mInputBuffer}, outputs);

        long t2 = SystemClock.elapsedRealtimeNanos();
        if (timings != null) timings.inferenceNs = t2 - t1;

        // 5. Parse proposals → filter confidence → NMS
        List<float[]> raw = parseProposals(mOutputDet[0], lb, roiLeft, roiTop, roiWidth, roiHeight);
        List<float[]> nms = applyNMS(raw);

        long t3 = SystemClock.elapsedRealtimeNanos();
        if (timings != null) timings.postprocessNs = t3 - t2;

        // 6. Decode mask + bangun objek Detection dengan timestamp lengkap
        long tMask0 = SystemClock.elapsedRealtimeNanos();
        List<Detection> result = buildDetections(nms, mOutputProto[0], lb, roiLeft, roiTop, roiWidth, roiHeight, captureTimestampNs);
        long tMask1 = SystemClock.elapsedRealtimeNanos();
        if (timings != null) timings.maskDecodeNs = tMask1 - tMask0;

        return result;
    }

    public List<Detection> detect(Bitmap screenBitmap,
                                  int roiLeft, int roiTop,
                                  int roiWidth, int roiHeight,
                                  StageTimings timings) {
        return detect(screenBitmap, roiLeft, roiTop, roiWidth, roiHeight, timings, SystemClock.elapsedRealtimeNanos());
    }

    public List<Detection> detect(Bitmap screenBitmap,
                                  int roiLeft, int roiTop,
                                  int roiWidth, int roiHeight) {
        return detect(screenBitmap, roiLeft, roiTop, roiWidth, roiHeight, null, SystemClock.elapsedRealtimeNanos());
    }

    // ─────────────────────────────────────────────────────────────
    // LETTERBOX
    // ─────────────────────────────────────────────────────────────

    private static class LetterboxResult {
        Bitmap bitmap;
        float  scale;    // scale yang sama untuk x dan y
        float  padLeft;
        float  padTop;
    }

    private LetterboxResult letterbox(Bitmap src, int size) {
        int srcW = src.getWidth();
        int srcH = src.getHeight();
        float scale = Math.min((float) size / srcW, (float) size / srcH);

        int newW = Math.round(srcW * scale);
        int newH = Math.round(srcH * scale);
        int padL = (size - newW) / 2;
        int padT = (size - newH) / 2;

        Bitmap canvas = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(canvas);
        c.drawColor(Color.BLACK);
        Bitmap scaled = Bitmap.createScaledBitmap(src, newW, newH, true);
        c.drawBitmap(scaled, padL, padT, null);
        scaled.recycle();

        LetterboxResult lb = new LetterboxResult();
        lb.bitmap  = canvas;
        lb.scale   = scale;
        lb.padLeft = padL;
        lb.padTop  = padT;
        return lb;
    }

    // ─────────────────────────────────────────────────────────────
    // PREPROCESSING: Bitmap → NCHW ByteBuffer
    // ─────────────────────────────────────────────────────────────

    private void encodeNCHW(Bitmap bmp) {
        int w = bmp.getWidth(), h = bmp.getHeight();
        bmp.getPixels(mPixels, 0, w, 0, 0, w, h);
        int n = w * h;
        for (int i = 0; i < n; i++) {
            int px   = mPixels[i];
            mChanR[i] = INV255[(px >> 16) & 0xFF];
            mChanG[i] = INV255[(px >>  8) & 0xFF];
            mChanB[i] = INV255[ px        & 0xFF];
        }
        mInputFloat.rewind();
        mInputFloat.put(mChanR, 0, n); // semua R
        mInputFloat.put(mChanG, 0, n); // semua G
        mInputFloat.put(mChanB, 0, n); // semua B
    }

    // ─────────────────────────────────────────────────────────────
    // PARSE PROPOSALS: output[37][2100] → list candidate box
    // ─────────────────────────────────────────────────────────────

    /**
     * Format output YOLOv8-Seg per proposal (kolom di output0):
     *   index 0: cx (model pixel space 0-320)
     *   index 1: cy
     *   index 2: w
     *   index 3: h
     *   index 4: confidence (sigmoid sudah dilakukan oleh model export)
     *   index 5..36: 32 mask coefficients
     *
     * Return format per entry float[]:
     *   [0..3]  = left, top, right, bottom  (screen pixels)
     *   [4,5]   = cx, cy                    (screen pixels)
     *   [6]     = confidence
     *   [7,8]   = leftRoi, topRoi           (ROI pixels, untuk mask clipping)
     *   [9,10]  = rightRoi, bottomRoi
     *   [11..42]= 32 mask coefficients
     */
    private List<float[]> parseProposals(float[][] det, LetterboxResult lb,
                                         int roiLeft, int roiTop,
                                         int roiWidth, int roiHeight) {
        List<float[]> out = new ArrayList<>();
        float scale   = lb.scale;
        float padLeft = lb.padLeft;
        float padTop  = lb.padTop;

        for (int i = 0; i < NUM_PROPOSALS; i++) {
            float conf = det[4][i];
            if (conf < CONF_THRESHOLD) continue;

            // Model output dalam piksel 0-320 (NCHW export Ultralytics)
            float cxM = det[0][i];
            float cyM = det[1][i];
            float wM  = det[2][i];
            float hM  = det[3][i];

            // Jika normalized (0-1), denormalisasi dulu
            if (cxM <= 1.0f && cyM <= 1.0f) {
                cxM *= INPUT_SIZE; cyM *= INPUT_SIZE;
                wM  *= INPUT_SIZE; hM  *= INPUT_SIZE;
            }

            // Letterbox inverse → ROI space
            float cxRoi = (cxM - padLeft) / scale;
            float cyRoi = (cyM - padTop)  / scale;
            float wRoi  = wM  / scale;
            float hRoi  = hM  / scale;

            if (wRoi < MIN_BOX_PX || hRoi < MIN_BOX_PX) continue;

            float lRoi = Math.max(0, cxRoi - wRoi / 2f);
            float tRoi = Math.max(0, cyRoi - hRoi / 2f);
            float rRoi = Math.min(roiWidth,  cxRoi + wRoi / 2f);
            float bRoi = Math.min(roiHeight, cyRoi + hRoi / 2f);

            float lScr = lRoi + roiLeft;
            float tScr = tRoi + roiTop;
            float rScr = rRoi + roiLeft;
            float bScr = bRoi + roiTop;
            float cxS  = (lScr + rScr) / 2f;
            float cyS  = (tScr + bScr) / 2f;

            float[] entry = new float[11 + NUM_MASK_COEFF];
            entry[0]  = lScr;   entry[1]  = tScr;
            entry[2]  = rScr;   entry[3]  = bScr;
            entry[4]  = cxS;    entry[5]  = cyS;
            entry[6]  = conf;
            entry[7]  = lRoi;   entry[8]  = tRoi;
            entry[9]  = rRoi;   entry[10] = bRoi;
            for (int k = 0; k < NUM_MASK_COEFF; k++) {
                entry[11 + k] = det[5 + k][i];
            }
            out.add(entry);
        }
        return out;
    }

    // ─────────────────────────────────────────────────────────────
    // NMS
    // ─────────────────────────────────────────────────────────────

    private List<float[]> applyNMS(List<float[]> boxes) {
        if (boxes.isEmpty()) return boxes;
        // Sort descending confidence
        boxes.sort((a, b) -> Float.compare(b[6], a[6]));

        List<float[]> kept       = new ArrayList<>();
        boolean[]     suppressed = new boolean[boxes.size()];

        for (int i = 0; i < boxes.size(); i++) {
            if (suppressed[i]) continue;
            kept.add(boxes.get(i));
            for (int j = i + 1; j < boxes.size(); j++) {
                if (suppressed[j]) continue;
                if (iou(boxes.get(i), boxes.get(j)) >= NMS_IOU_THRESH) suppressed[j] = true;
            }
        }
        return kept;
    }

    private float iou(float[] a, float[] b) {
        float iL = Math.max(a[0], b[0]), iT = Math.max(a[1], b[1]);
        float iR = Math.min(a[2], b[2]), iB = Math.min(a[3], b[3]);
        float iW = Math.max(0, iR - iL), iH = Math.max(0, iB - iT);
        float inter = iW * iH;
        float areaA = (a[2]-a[0]) * (a[3]-a[1]);
        float areaB = (b[2]-b[0]) * (b[3]-b[1]);
        float uni   = areaA + areaB - inter;
        return uni <= 0 ? 0f : inter / uni;
    }

    // ─────────────────────────────────────────────────────────────
    // BUILD DETECTIONS + MASK DECODE
    // ─────────────────────────────────────────────────────────────

    private List<Detection> buildDetections(List<float[]> boxes,
                                            float[][][] proto,
                                            LetterboxResult lb,
                                            int roiLeft, int roiTop,
                                            int roiWidth, int roiHeight,
                                            long captureTimestampNs) {
        List<Detection> result = new ArrayList<>();
        long nowNs = SystemClock.elapsedRealtimeNanos();

        for (float[] entry : boxes) {
            Detection det = new Detection();
            det.left       = entry[0];
            det.top        = entry[1];
            det.right      = entry[2];
            det.bottom     = entry[3];
            det.cx         = entry[4];
            det.cy         = entry[5];
            det.confidence = entry[6];
            det.roiLeft    = roiLeft;
            det.roiTop     = roiTop;
            det.roiWidth   = roiWidth;
            det.roiHeight  = roiHeight;

            // Simpan posisi RAW YOLO asli sebelum proses smoothing EMA
            det.rawLeft       = det.left;
            det.rawTop        = det.top;
            det.rawRight      = det.right;
            det.rawBottom     = det.bottom;
            det.rawCx         = det.cx;
            det.rawCy         = det.cy;
            det.rawConfidence = det.confidence;

            // Metadata temporal & status
            det.captureTimestampNs      = captureTimestampNs;
            det.inferenceEndTimestampNs = nowNs;
            det.isFresh                 = true;
            det.holdAgeMs               = 0.0;
            det.isStale                 = false;
            det.displacementPx          = 0f;
            det.currentAlpha            = 1.0f;

            float cyRel = det.cy - roiTop;
            det.depthMm = Math.max(0f, Math.min((cyRel / roiHeight) * DEPTH_SCALE_MM, DEPTH_SCALE_MM));

            // Decode instance mask: coeffs[32] × proto[32][80][80] → sigmoid → threshold
            float[] coeffs = new float[NUM_MASK_COEFF];
            for (int k = 0; k < NUM_MASK_COEFF; k++) coeffs[k] = entry[11 + k];
            det.maskBitmap = decodeMask(coeffs, proto, lb, entry[7], entry[8], entry[9], entry[10]);

            result.add(det);
        }
        return result;
    }

    /**
     * Decode instance mask untuk satu deteksi.
     *
     * Langkah:
     *   1. Dot product: raw[py][px] = Σ (coeffs[k] * proto[k][py][px])
     *   2. Sigmoid: sig = 1 / (1 + exp(-raw))
     *   3. Threshold: jika sig > MASK_THRESHOLD → warna vena (biru semi-transparan)
     *   4. Clip ke area bbox (kurangi false positive di luar bbox)
     *
     * Hasilnya adalah Bitmap 80×80 yang kemudian akan di-scale ke bbox di OverlayView.
     */
    /**
     * Decode instance mask untuk satu deteksi dengan perbaikan cropped bbox & exterior contour.
     *
     * Perbaikan Matematis:
     *   1. Bitmap mask dibuat persis seukuran bounding box dalam prototype space (cropW × cropH).
     *      Dengan ukuran pas ini, saat OverlayView meregangkan bitmap ke mMaskDst (bbox layar),
     *      mask terpetakan 1:1 tanpa penyusutan 97.6% seperti pada implementasi lama.
     *   2. Ekstraksi exterior contour: piksel tepi vena diwarnai dengan kontur tajam ber-kontras
     *      tinggi, sedangkan bagian dalam lumen diisi warna semi-transparan.
     */
    private Bitmap decodeMask(float[] coeffs, float[][][] proto,
                               LetterboxResult lb,
                               float lRoi, float tRoi, float rRoi, float bRoi) {
        // Konversi bbox ROI → proto space (80×80)
        float protoScale = (float) PROTO_H / INPUT_SIZE;
        int clipL = Math.max(0, Math.min(PROTO_W - 1, (int)((lRoi * lb.scale + lb.padLeft) * protoScale)));
        int clipT = Math.max(0, Math.min(PROTO_H - 1, (int)((tRoi * lb.scale + lb.padTop)  * protoScale)));
        int clipR = Math.max(0, Math.min(PROTO_W - 1, (int)((rRoi * lb.scale + lb.padLeft) * protoScale)));
        int clipB = Math.max(0, Math.min(PROTO_H - 1, (int)((bRoi * lb.scale + lb.padTop)  * protoScale)));

        if (clipR < clipL) clipR = clipL;
        if (clipB < clipT) clipB = clipT;

        int cropW = clipR - clipL + 1;
        int cropH = clipB - clipT + 1;
        boolean[][] isVein = new boolean[cropH][cropW];

        for (int py = clipT; py <= clipB; py++) {
            int yIdx = py - clipT;
            for (int px = clipL; px <= clipR; px++) {
                int xIdx = px - clipL;
                // Dot product: Σ coeffs[k] * proto[k][py][px]
                float val = 0f;
                for (int k = 0; k < NUM_MASK_COEFF; k++) {
                    val += coeffs[k] * proto[k][py][px];
                }
                // Sigmoid
                float sig = 1f / (1f + (float) Math.exp(-val));
                isVein[yIdx][xIdx] = (sig > MASK_THRESHOLD);
            }
        }

        // Warna lumen (interior): Cyan semi-transparan
        final int COLOR_INTERIOR = Color.argb(125, 0, 180, 255);
        // Warna kontur luar (exterior contour): Cyan-putih terang berdaya kontras tinggi
        final int COLOR_CONTOUR  = Color.argb(235, 0, 255, 235);

        int[] cropPixels = new int[cropW * cropH];
        for (int y = 0; y < cropH; y++) {
            int yOff = y * cropW;
            for (int x = 0; x < cropW; x++) {
                if (isVein[y][x]) {
                    // Deteksi pixel tepi (exterior boundary)
                    boolean isEdge = (x == 0 || x == cropW - 1 || y == 0 || y == cropH - 1 ||
                                      !isVein[y - 1][x] || !isVein[y + 1][x] ||
                                      !isVein[y][x - 1] || !isVein[y][x + 1]);
                    cropPixels[yOff + x] = isEdge ? COLOR_CONTOUR : COLOR_INTERIOR;
                } else {
                    cropPixels[yOff + x] = Color.TRANSPARENT;
                }
            }
        }

        return Bitmap.createBitmap(cropPixels, cropW, cropH, Bitmap.Config.ARGB_8888);
    }

    // ─────────────────────────────────────────────────────────────
    // RESOURCE MANAGEMENT
    // ─────────────────────────────────────────────────────────────

    public void close() {
        if (mInterpreter != null) { mInterpreter.close(); mInterpreter = null; }
        if (mGpuDelegate  != null) { mGpuDelegate.close();  mGpuDelegate  = null; }
        Log.d(TAG, "YoloDetector closed");
    }
}
