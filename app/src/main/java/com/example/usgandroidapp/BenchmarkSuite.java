package com.example.usgandroidapp;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;

import java.io.File;
import java.io.FileWriter;
import java.io.InputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * BenchmarkSuite.java
 * ====================
 * Suite pengujian performa runtime TFLite & evaluasi konfigurasi eksekusi
 * pada perangkat target Android (Samsung Galaxy Tab A 8.0 SM-T295).
 */
public class BenchmarkSuite {

    private static final String TAG = "TFLiteBenchmark";

    public static final int WARMUP_FRAMES = 30;
    public static final int MEASURE_FRAMES = 200;

    // Dimensi layar simulasi (sesuai tablet Samsung SM-T295)
    private static final int SCREEN_W = 1280;
    private static final int SCREEN_H = 800;

    // ROI koordinat
    private static final int ROI_LEFT   = (int) (SCREEN_W * 0.20f);
    private static final int ROI_TOP    = (int) (SCREEN_H * 0.07f);
    private static final int ROI_WIDTH  = (int) (SCREEN_W * 0.60f);
    private static final int ROI_HEIGHT = (int) (SCREEN_H * 0.45f);

    // Adaptive EMA konstanta (identik dengan VeinDetectionService)
    private static final float ALPHA_STILL    = 0.20f;
    private static final float ALPHA_FAST     = 1.00f;
    private static final float MOVE_THRESH_PX = 12f;
    private static final float JUMP_THRESH_PX = 70f;
    private static final int   MAX_HOLD       = 6;

    /**
     * Konfigurasi eksperimen
     */
    public static class ExperimentDef {
        public String id;
        public String name;
        public String modelPath;
        public int numThreads;
        public boolean useXnnpack;
        public boolean enableNnapi;
        public boolean enableGpu;

        public ExperimentDef(String id, String name, String modelPath, int numThreads, boolean useXnnpack, boolean enableNnapi, boolean enableGpu) {
            this.id = id;
            this.name = name;
            this.modelPath = modelPath;
            this.numThreads = numThreads;
            this.useXnnpack = useXnnpack;
            this.enableNnapi = enableNnapi;
            this.enableGpu = enableGpu;
        }
    }

    /**
     * Hasil pengukuran satu konfigurasi
     */
    public static class BenchmarkResult {
        public ExperimentDef config;
        public int repetition = 1;
        public double initTimeMs;
        public String runtimeDelegate;
        public long modelSizeBytes;
        public int measuredFrames;

        // Monitoring hardware
        public float batteryTempStartC = -1f;
        public float batteryTempEndC   = -1f;
        public long pssMemMb           = -1;

        // Stage stats (ms)
        public PerformanceProfiler.StageStats sPreprocess;
        public PerformanceProfiler.StageStats sInference;
        public PerformanceProfiler.StageStats sPostprocess;
        public PerformanceProfiler.StageStats sMaskDecode;
        public PerformanceProfiler.StageStats sEma;
        public PerformanceProfiler.StageStats sPipeline;

        public double wallClockSec;
        public double actualFps;
        public double theoreticalFps;

        // Agreement & detection stats
        public int detectedCount;
        public double avgConfidence;
        public double meanCx;
        public double meanCy;
        public double meanW;
        public double meanH;
        public double meanIouWithBaseline;

        // Simpan prediksi bbox per frame untuk evaluasi agreement
        public float[][] bboxes = new float[MEASURE_FRAMES][4]; // [left, top, right, bottom]
        public boolean[] hasDetections = new boolean[MEASURE_FRAMES];
    }

    public static float getBatteryTemperature(Context context) {
        try {
            android.content.IntentFilter ifilter = new android.content.IntentFilter(android.content.Intent.ACTION_BATTERY_CHANGED);
            android.content.Intent b = context.registerReceiver(null, ifilter);
            if (b != null) {
                int t = b.getIntExtra(android.os.BatteryManager.EXTRA_TEMPERATURE, -1);
                if (t > 0) return t / 10.0f;
            }
        } catch (Exception ignored) {}
        return -1f;
    }

    public static long getPssMemMb() {
        try {
            return android.os.Debug.getPss() / 1024;
        } catch (Exception ignored) {}
        return -1;
    }

    /**
     * Jalankan seluruh rangkaian eksperimen.
     */
    public static List<BenchmarkResult> runAll(Context context) {
        Log.i(TAG, "==========================================================================");
        Log.i(TAG, "        MEMULAI EXPERIMENTAL BENCHMARK SUITE — RUNTIME TFLITE             ");
        Log.i(TAG, "==========================================================================");
        Log.i(TAG, "Perangkat: " + Build.MANUFACTURER + " " + Build.MODEL + " (Board: " + Build.BOARD + ")");
        Log.i(TAG, "Hardware : " + Build.HARDWARE + " | Android " + Build.VERSION.RELEASE);
        Log.i(TAG, "Warm-up  : " + WARMUP_FRAMES + " frames | Terukur: " + MEASURE_FRAMES + " frames per config");
        Log.i(TAG, "==========================================================================");

        List<Bitmap> testBitmaps = loadTestFrames(context);
        Log.i(TAG, "Loaded " + testBitmaps.size() + " test frames untuk benchmark.");

        List<ExperimentDef> experiments = new ArrayList<>();
        // Config A: Baseline
        experiments.add(new ExperimentDef("A", "FP32 Baseline (CPU 4T, XNNPACK)", "YoloV8_320.tflite", 4, true, false, false));
        // Config B1: 3 Threads
        experiments.add(new ExperimentDef("B1", "FP32 CPU 3T (XNNPACK)", "YoloV8_320.tflite", 3, true, false, false));
        // Config B2: 2 Threads
        experiments.add(new ExperimentDef("B2", "FP32 CPU 2T (XNNPACK)", "YoloV8_320.tflite", 2, true, false, false));
        // Config B3: No-XNNPACK (Reference CPU)
        experiments.add(new ExperimentDef("B3", "FP32 CPU 4T (No-XNNPACK / Ref)", "YoloV8_320.tflite", 4, false, false, false));
        // Config B4: Model Variant YoloV26_320 (4T)
        experiments.add(new ExperimentDef("B4", "FP32 YoloV26 (CPU 4T, XNNPACK)", "YoloV26_320.tflite", 4, true, false, false));
        // Config B5: Model Variant YoloV26_320 (3T)
        experiments.add(new ExperimentDef("B5", "FP32 YoloV26 (CPU 3T, XNNPACK)", "YoloV26_320.tflite", 3, true, false, false));
        // Config B6: NNAPI Delegate
        experiments.add(new ExperimentDef("B6", "FP32 NNAPI Delegate", "YoloV8_320.tflite", 4, true, true, false));

        List<BenchmarkResult> results = new ArrayList<>();
        BenchmarkResult baselineResult = null;

        for (ExperimentDef exp : experiments) {
            try {
                Log.i(TAG, "\n>>> Menjalankan Eksperimen [" + exp.id + "]: " + exp.name);
                BenchmarkResult res = runSingleExperiment(context, exp, testBitmaps);
                if (res != null) {
                    if (baselineResult == null) {
                        baselineResult = res;
                        res.meanIouWithBaseline = 1.0;
                    } else {
                        res.meanIouWithBaseline = computeAgreementIoU(baselineResult, res);
                    }
                    results.add(res);
                }
            } catch (Throwable t) {
                Log.e(TAG, "Gagal menjalankan eksperimen " + exp.id + ": " + t.getMessage(), t);
            }

            // Istirahat 3 detik antar eksperimen untuk stabilisasi termal
            try { Thread.sleep(3000); } catch (InterruptedException ignored) {}
        }

        // Hasilkan laporan akhir
        exportResults(context, results);

        // Bersihkan bitmap
        for (Bitmap b : testBitmaps) {
            if (b != null && !b.isRecycled()) b.recycle();
        }

        return results;
    }

    private static BenchmarkResult runSingleExperiment(Context context, ExperimentDef exp, List<Bitmap> testBitmaps) {
        YoloDetector.RuntimeConfig cfg = new YoloDetector.RuntimeConfig();
        cfg.modelPath   = exp.modelPath;
        cfg.numThreads  = exp.numThreads;
        cfg.useXnnpack  = exp.useXnnpack;
        cfg.enableNnapi = exp.enableNnapi;
        cfg.enableGpu   = exp.enableGpu;

        long tInitStart = SystemClock.elapsedRealtimeNanos();
        YoloDetector detector;
        try {
            detector = new YoloDetector(context, cfg);
        } catch (Exception e) {
            Log.e(TAG, "Init model gagal untuk " + exp.id + ": " + e.getMessage());
            return null;
        }
        long tInitEnd = SystemClock.elapsedRealtimeNanos();
        double initTimeMs = (tInitEnd - tInitStart) / 1_000_000.0;

        BenchmarkResult res = new BenchmarkResult();
        res.config = exp;
        res.initTimeMs = initTimeMs;
        res.runtimeDelegate = detector.getDelegateName();
        res.measuredFrames = MEASURE_FRAMES;
        res.batteryTempStartC = getBatteryTemperature(context);

        // Ambil ukuran model di assets
        try {
            android.content.res.AssetFileDescriptor fd = context.getAssets().openFd(exp.modelPath);
            res.modelSizeBytes = fd.getLength();
            fd.close();
        } catch (Exception ignored) {
            res.modelSizeBytes = 0;
        }

        Log.i(TAG, String.format(Locale.US, "Model %s initialized dalam %.2f ms (Delegate: %s, Size: %.2f MB)",
                exp.modelPath, initTimeMs, res.runtimeDelegate, res.modelSizeBytes / (1024.0 * 1024.0)));

        // State EMA
        float smL = -1f, smT = -1f, smR = -1f, smB = -1f;
        float smCx = -1f, smCy = -1f;
        YoloDetector.Detection lastValid = null;
        int holdCount = 0;

        YoloDetector.StageTimings timings = new YoloDetector.StageTimings();

        // 1. Fase Warm-up (30 frame)
        Log.i(TAG, "Menjalankan warm-up (" + WARMUP_FRAMES + " frames)...");
        for (int i = 0; i < WARMUP_FRAMES; i++) {
            Bitmap bmp = testBitmaps.get(i % testBitmaps.size());
            List<YoloDetector.Detection> dets = detector.detect(bmp, ROI_LEFT, ROI_TOP, ROI_WIDTH, ROI_HEIGHT, timings);
            if (dets != null) {
                for (YoloDetector.Detection d : dets) {
                    if (d.maskBitmap != null && !d.maskBitmap.isRecycled()) d.maskBitmap.recycle();
                }
            }
        }
        Log.i(TAG, "Warm-up selesai. Memulai pengukuran steady-state (" + MEASURE_FRAMES + " frames)...");

        // 2. Steady-state measurement
        long[] preNs  = new long[MEASURE_FRAMES];
        long[] infNs  = new long[MEASURE_FRAMES];
        long[] postNs = new long[MEASURE_FRAMES];
        long[] maskNs = new long[MEASURE_FRAMES];
        long[] emaNs  = new long[MEASURE_FRAMES];
        long[] pipeNs = new long[MEASURE_FRAMES];

        int detectedCount = 0;
        double sumConf = 0.0;
        double sumCx = 0.0, sumCy = 0.0, sumW = 0.0, sumH = 0.0;

        long tBenchmarkStart = SystemClock.elapsedRealtimeNanos();

        for (int i = 0; i < MEASURE_FRAMES; i++) {
            Bitmap bmp = testBitmaps.get(i % testBitmaps.size());
            long tPipeStart = SystemClock.elapsedRealtimeNanos();

            List<YoloDetector.Detection> rawDets = detector.detect(bmp, ROI_LEFT, ROI_TOP, ROI_WIDTH, ROI_HEIGHT, timings);

            // EMA stage
            long tEmaStart = SystemClock.elapsedRealtimeNanos();
            YoloDetector.Detection best = (rawDets != null && !rawDets.isEmpty()) ? rawDets.get(0) : null;
            YoloDetector.Detection smoothedDet = null;

            if (best != null) {
                if (smL < 0) {
                    smL = best.left;  smT = best.top;
                    smR = best.right; smB = best.bottom;
                    smCx = best.cx;   smCy = best.cy;
                } else {
                    float dx = best.cx - smCx;
                    float dy = best.cy - smCy;
                    float disp = (float) Math.sqrt(dx * dx + dy * dy);
                    float t = Math.min(1f, Math.max(0f, (disp - MOVE_THRESH_PX) / (JUMP_THRESH_PX - MOVE_THRESH_PX)));
                    float alpha = ALPHA_STILL + t * (ALPHA_FAST - ALPHA_STILL);

                    smL  += alpha * (best.left   - smL);
                    smT  += alpha * (best.top    - smT);
                    smR  += alpha * (best.right  - smR);
                    smB  += alpha * (best.bottom - smB);
                    smCx += alpha * (best.cx     - smCx);
                    smCy += alpha * (best.cy     - smCy);
                }
                best.left = smL;  best.top = smT;
                best.right = smR; best.bottom = smB;
                best.cx = smCx;   best.cy = smCy;
                lastValid = best;
                holdCount = MAX_HOLD;
                smoothedDet = best;
            } else {
                if (holdCount > 0 && lastValid != null) {
                    holdCount--;
                    smoothedDet = lastValid;
                } else {
                    smL = -1f;
                    lastValid = null;
                }
            }
            long tEmaEnd = SystemClock.elapsedRealtimeNanos();
            long tPipeEnd = SystemClock.elapsedRealtimeNanos();

            preNs[i]  = timings.preprocessNs;
            infNs[i]  = timings.inferenceNs;
            postNs[i] = timings.postprocessNs;
            maskNs[i] = timings.maskDecodeNs;
            emaNs[i]  = tEmaEnd - tEmaStart;
            pipeNs[i] = tPipeEnd - tPipeStart;

            if (smoothedDet != null) {
                detectedCount++;
                sumConf += smoothedDet.confidence;
                sumCx += smoothedDet.cx;
                sumCy += smoothedDet.cy;
                sumW  += (smoothedDet.right - smoothedDet.left);
                sumH  += (smoothedDet.bottom - smoothedDet.top);

                res.hasDetections[i] = true;
                res.bboxes[i][0] = smoothedDet.left;
                res.bboxes[i][1] = smoothedDet.top;
                res.bboxes[i][2] = smoothedDet.right;
                res.bboxes[i][3] = smoothedDet.bottom;
            } else {
                res.hasDetections[i] = false;
            }

            // Bersihkan mask bitmap
            if (rawDets != null) {
                for (YoloDetector.Detection d : rawDets) {
                    if (d.maskBitmap != null && !d.maskBitmap.isRecycled()) d.maskBitmap.recycle();
                }
            }

            if ((i + 1) % 50 == 0) {
                Log.d(TAG, String.format(Locale.US, "  [%s] Progress: %d/%d (Pipe: %.2f ms, Model: %.2f ms)",
                        exp.id, (i + 1), MEASURE_FRAMES, pipeNs[i] / 1_000_000.0, infNs[i] / 1_000_000.0));
            }
        }

        long tBenchmarkEnd = SystemClock.elapsedRealtimeNanos();
        res.wallClockSec = (tBenchmarkEnd - tBenchmarkStart) / 1_000_000_000.0;
        res.actualFps = (res.wallClockSec > 0) ? (MEASURE_FRAMES / res.wallClockSec) : 0;

        res.sPreprocess  = computeStats("Preprocessing", preNs);
        res.sInference   = computeStats("TFLite Model", infNs);
        res.sPostprocess = computeStats("Parsing + NMS", postNs);
        res.sMaskDecode  = computeStats("Mask Decode", maskNs);
        res.sEma         = computeStats("Adaptive EMA", emaNs);
        res.sPipeline    = computeStats("Pipeline Total", pipeNs);

        res.theoreticalFps = (res.sPipeline.medianMs > 0) ? (1000.0 / res.sPipeline.medianMs) : 0;

        res.detectedCount = detectedCount;
        if (detectedCount > 0) {
            res.avgConfidence = sumConf / detectedCount;
            res.meanCx = sumCx / detectedCount;
            res.meanCy = sumCy / detectedCount;
            res.meanW  = sumW  / detectedCount;
            res.meanH  = sumH  / detectedCount;
        }

        detector.close();
        res.batteryTempEndC = getBatteryTemperature(context);
        res.pssMemMb        = getPssMemMb();

        Log.i(TAG, String.format(Locale.US,
                ">>> Hasil [%s]: Med Model=%.2f ms | Med Pipe=%.2f ms | FPS=%.2f (Wall: %.2f s | Temp: %.1fC -> %.1fC | RAM: %d MB)",
                exp.id, res.sInference.medianMs, res.sPipeline.medianMs, res.actualFps, res.wallClockSec,
                res.batteryTempStartC, res.batteryTempEndC, res.pssMemMb));

        return res;
    }

    private static double computeAgreementIoU(BenchmarkResult baseline, BenchmarkResult other) {
        int countBoth = 0;
        double sumIoU = 0.0;
        for (int i = 0; i < MEASURE_FRAMES; i++) {
            if (baseline.hasDetections[i] && other.hasDetections[i]) {
                float[] bA = baseline.bboxes[i];
                float[] bB = other.bboxes[i];

                float xA = Math.max(bA[0], bB[0]);
                float yA = Math.max(bA[1], bB[1]);
                float xB = Math.min(bA[2], bB[2]);
                float yB = Math.min(bA[3], bB[3]);

                float interArea = Math.max(0, xB - xA) * Math.max(0, yB - yA);
                float areaA = Math.max(0, bA[2] - bA[0]) * Math.max(0, bA[3] - bA[1]);
                float areaB = Math.max(0, bB[2] - bB[0]) * Math.max(0, bB[3] - bB[1]);
                float unionArea = areaA + areaB - interArea;

                if (unionArea > 0) {
                    sumIoU += (interArea / unionArea);
                    countBoth++;
                }
            }
        }
        return (countBoth > 0) ? (sumIoU / countBoth) : 0.0;
    }

    private static PerformanceProfiler.StageStats computeStats(String name, long[] rawNs) {
        PerformanceProfiler.StageStats s = new PerformanceProfiler.StageStats();
        s.name = name;
        s.count = rawNs.length;
        if (s.count == 0) return s;

        double[] ms = new double[s.count];
        double sum = 0.0;
        for (int i = 0; i < s.count; i++) {
            double v = rawNs[i] / 1_000_000.0;
            ms[i] = v;
            sum += v;
        }
        Arrays.sort(ms);

        s.minMs  = ms[0];
        s.maxMs  = ms[s.count - 1];
        s.meanMs = sum / s.count;
        s.medianMs = (s.count % 2 == 1) ? ms[s.count / 2] : ((ms[s.count / 2 - 1] + ms[s.count / 2]) / 2.0);

        int p95Idx = (int) Math.ceil(0.95 * s.count) - 1;
        if (p95Idx < 0) p95Idx = 0;
        if (p95Idx >= s.count) p95Idx = s.count - 1;
        s.p95Ms = ms[p95Idx];
        return s;
    }

    private static List<Bitmap> loadTestFrames(Context context) {
        List<Bitmap> bitmaps = new ArrayList<>();
        AssetManager am = context.getAssets();
        String[] sampleNames = {"benchmark_frames/frame_01.png", "benchmark_frames/frame_02.png",
                "benchmark_frames/frame_03.png", "benchmark_frames/frame_04.png", "benchmark_frames/frame_05.png"};

        for (String name : sampleNames) {
            try (InputStream is = am.open(name)) {
                Bitmap decoded = BitmapFactory.decodeStream(is);
                if (decoded != null) {
                    // Masukkan ke kanvas ukuran layar tablet (1280x800)
                    Bitmap screenBmp = Bitmap.createBitmap(SCREEN_W, SCREEN_H, Bitmap.Config.ARGB_8888);
                    Canvas canvas = new Canvas(screenBmp);
                    canvas.drawColor(Color.BLACK);
                    // Gambar USG di area tengah layar (area ROI)
                    android.graphics.Rect src = new android.graphics.Rect(0, 0, decoded.getWidth(), decoded.getHeight());
                    android.graphics.Rect dst = new android.graphics.Rect(ROI_LEFT, ROI_TOP, ROI_LEFT + ROI_WIDTH, ROI_TOP + ROI_HEIGHT);
                    canvas.drawBitmap(decoded, src, dst, null);
                    bitmaps.add(screenBmp);
                    decoded.recycle();
                }
            } catch (Exception e) {
                Log.w(TAG, "Gagal memuat asset " + name + ": " + e.getMessage());
            }
        }

        if (bitmaps.isEmpty()) {
            Log.w(TAG, "Asset USG tidak ditemukan, membuat synthetic USG frame...");
            Bitmap syn = Bitmap.createBitmap(SCREEN_W, SCREEN_H, Bitmap.Config.ARGB_8888);
            Canvas canvas = new Canvas(syn);
            canvas.drawColor(Color.rgb(20, 20, 20));
            Paint p = new Paint();
            p.setColor(Color.rgb(10, 10, 10));
            canvas.drawOval(ROI_LEFT + ROI_WIDTH * 0.4f, ROI_TOP + ROI_HEIGHT * 0.4f,
                    ROI_LEFT + ROI_WIDTH * 0.6f, ROI_TOP + ROI_HEIGHT * 0.6f, p);
            bitmaps.add(syn);
        }

        return bitmaps;
    }

    private static void exportResults(Context context, List<BenchmarkResult> results) {
        File dir = context.getExternalFilesDir(null);
        if (dir == null) dir = context.getFilesDir();

        File csvFile = new File(dir, "tflite_benchmark_results.csv");
        File mdFile  = new File(dir, "tflite_benchmark_summary.md");

        try (PrintWriter pw = new PrintWriter(new FileWriter(csvFile))) {
            pw.println("config_id,config_name,model_path,model_size_mb,threads,xnnpack,delegate,init_ms," +
                    "prep_med_ms,model_med_ms,post_med_ms,mask_med_ms,ema_med_ms,pipe_med_ms,pipe_mean_ms,pipe_p95_ms," +
                    "actual_fps,theo_fps,wall_sec,detected_pct,avg_conf,iou_baseline");

            for (BenchmarkResult r : results) {
                pw.printf(Locale.US,
                        "%s,\"%s\",%s,%.2f,%d,%b,\"%s\",%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.3f,%.3f\n",
                        r.config.id, r.config.name, r.config.modelPath, r.modelSizeBytes / (1024.0 * 1024.0),
                        r.config.numThreads, r.config.useXnnpack, r.runtimeDelegate, r.initTimeMs,
                        r.sPreprocess.medianMs, r.sInference.medianMs, r.sPostprocess.medianMs,
                        r.sMaskDecode.medianMs, r.sEma.medianMs, r.sPipeline.medianMs,
                        r.sPipeline.meanMs, r.sPipeline.p95Ms, r.actualFps, r.theoreticalFps,
                        r.wallClockSec, (r.detectedCount * 100.0 / r.measuredFrames), r.avgConfidence,
                        r.meanIouWithBaseline);
            }
            Log.i(TAG, "CSV benchmark berhasil disimpan di: " + csvFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e(TAG, "Gagal simpan CSV: " + e.getMessage());
        }

        // Print Formatted Table to Logcat
        StringBuilder sb = new StringBuilder();
        sb.append("\n=========================================================================================================\n");
        sb.append("                       RINGKASAN BENCHMARK RUNTIME TFLITE PADA SAMSUNG SM-T295                           \n");
        sb.append("=========================================================================================================\n");
        sb.append(String.format("%-4s | %-28s | %-6s | %-7s | %-8s | %-8s | %-6s | %-6s | %-6s\n",
                "ID", "Konfigurasi", "SizeMB", "Init ms", "Model ms", "Pipe ms", "FPS", "IoU-BL", "Conf"));
        sb.append("---------------------------------------------------------------------------------------------------------\n");
        for (BenchmarkResult r : results) {
            sb.append(String.format(Locale.US,
                    "%-4s | %-28s | %6.2f | %7.1f | %8.2f | %8.2f | %6.2f | %6.3f | %6.3f\n",
                    r.config.id, r.config.name, r.modelSizeBytes / (1024.0 * 1024.0), r.initTimeMs,
                    r.sInference.medianMs, r.sPipeline.medianMs, r.actualFps, r.meanIouWithBaseline, r.avgConfidence));
        }
        sb.append("=========================================================================================================\n");

        Log.i(TAG, sb.toString());
    }

    /**
     * Jalankan suite reproduktibilitas: membandingkan Config A dan Config B3
     * masing-masing 3 pengulangan dengan urutan berselang-seling dan pendinginan termal.
     */
    public static List<BenchmarkResult> runReproducibilitySuite(Context context) {
        final String REP_TAG = "TFLiteReproducibility";
        Log.i(REP_TAG, "==========================================================================");
        Log.i(REP_TAG, "      MEMULAI BENCHMARK REPRODUCIBILITY (CONFIG A vs CONFIG B3)           ");
        Log.i(REP_TAG, "==========================================================================");
        Log.i(REP_TAG, "Perangkat: " + Build.MANUFACTURER + " " + Build.MODEL + " (Hardware: " + Build.HARDWARE + ")");
        Log.i(REP_TAG, "Protokol : 3 Pengulangan (R1: A->B3, R2: B3->A, R3: A->B3)");
        Log.i(REP_TAG, "Warm-up  : " + WARMUP_FRAMES + " frame | Terukur: " + MEASURE_FRAMES + " frame per run");
        Log.i(REP_TAG, "==========================================================================");

        List<Bitmap> testBitmaps = loadTestFrames(context);

        ExperimentDef expA  = new ExperimentDef("A", "FP32 CPU 4T (XNNPACK)", "YoloV8_320.tflite", 4, true, false, false);
        ExperimentDef expB3 = new ExperimentDef("B3", "FP32 CPU 4T (No-XNNPACK)", "YoloV8_320.tflite", 4, false, false, false);

        class RunPlan {
            int rep;
            ExperimentDef exp;
            RunPlan(int r, ExperimentDef e) { this.rep = r; this.exp = e; }
        }
        List<RunPlan> plan = new ArrayList<>();
        plan.add(new RunPlan(1, expA));
        plan.add(new RunPlan(1, expB3));
        plan.add(new RunPlan(2, expB3));
        plan.add(new RunPlan(2, expA));
        plan.add(new RunPlan(3, expA));
        plan.add(new RunPlan(3, expB3));

        List<BenchmarkResult> results = new ArrayList<>();
        BenchmarkResult baselineA = null;

        for (int i = 0; i < plan.size(); i++) {
            RunPlan p = plan.get(i);
            Log.i(REP_TAG, String.format("\n>>> [%d/6] Repetisi %d — Menjalankan Config %s (%s)",
                    (i + 1), p.rep, p.exp.id, p.exp.name));

            BenchmarkResult r = runSingleExperiment(context, p.exp, testBitmaps);
            if (r != null) {
                r.repetition = p.rep;
                if (baselineA == null && "A".equals(p.exp.id)) {
                    baselineA = r;
                    r.meanIouWithBaseline = 1.0;
                } else if (baselineA != null) {
                    r.meanIouWithBaseline = computeAgreementIoU(baselineA, r);
                }
                results.add(r);
            }

            // Pendinginan termal 5 detik antarpengujian
            if (i < plan.size() - 1) {
                Log.i(REP_TAG, "Pendinginan termal 5 detik sebelum run berikutnya...");
                try { Thread.sleep(5000); } catch (InterruptedException ignored) {}
            }
        }

        // Export CSV & Log table
        exportReproducibilityResults(context, results);

        // Bersihkan bitmap
        for (Bitmap b : testBitmaps) {
            if (b != null && !b.isRecycled()) b.recycle();
        }

        return results;
    }

    private static void exportReproducibilityResults(Context context, List<BenchmarkResult> results) {
        File dir = context.getExternalFilesDir(null);
        if (dir == null) dir = context.getFilesDir();

        File csvFile = new File(dir, "reproducibility_benchmark_results.csv");

        try (PrintWriter pw = new PrintWriter(new FileWriter(csvFile))) {
            pw.println("repetition,config_id,config_name,xnnpack,threads,init_ms,model_med_ms,model_mean_ms,model_p95_ms,pipe_med_ms,pipe_p95_ms,actual_fps,wall_sec,temp_start_c,temp_end_c,pss_mb,iou_baseline");

            for (BenchmarkResult r : results) {
                pw.printf(Locale.US,
                        "%d,%s,\"%s\",%b,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.1f,%.1f,%d,%.3f\n",
                        r.repetition, r.config.id, r.config.name, r.config.useXnnpack, r.config.numThreads,
                        r.initTimeMs, r.sInference.medianMs, r.sInference.meanMs, r.sInference.p95Ms,
                        r.sPipeline.medianMs, r.sPipeline.p95Ms, r.actualFps, r.wallClockSec,
                        r.batteryTempStartC, r.batteryTempEndC, r.pssMemMb, r.meanIouWithBaseline);
            }
            Log.i("TFLiteReproducibility", "CSV Reproducibility tersimpan di: " + csvFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e("TFLiteReproducibility", "Gagal simpan CSV: " + e.getMessage());
        }

        // Print Formatted Summary
        StringBuilder sb = new StringBuilder();
        sb.append("\n======================================================================================================================\n");
        sb.append("                       HASIL UJI REPRODUCIBILITY 3 PENGULANGAN (CONFIG A vs CONFIG B3)                                \n");
        sb.append("======================================================================================================================\n");
        sb.append(String.format("%-4s | %-4s | %-22s | %-7s | %-8s | %-8s | %-8s | %-6s | %-12s | %-6s\n",
                "Rep", "ID", "Konfigurasi", "Init ms", "Model ms", "Pipe ms", "p95 ms", "FPS", "Suhu Baterai", "RAM"));
        sb.append("----------------------------------------------------------------------------------------------------------------------\n");
        for (BenchmarkResult r : results) {
            sb.append(String.format(Locale.US,
                    "%-4d | %-4s | %-22s | %7.1f | %8.2f | %8.2f | %8.2f | %6.2f | %4.1fC -> %4.1fC | %4dMB\n",
                    r.repetition, r.config.id, r.config.name, r.initTimeMs,
                    r.sInference.medianMs, r.sPipeline.medianMs, r.sPipeline.p95Ms, r.actualFps,
                    r.batteryTempStartC, r.batteryTempEndC, r.pssMemMb));
        }
        sb.append("======================================================================================================================\n");

        Log.i("TFLiteReproducibility", sb.toString());
    }
}

