package com.example.usgandroidapp;

import android.content.Context;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;

import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * PerformanceProfiler.java
 * ========================
 * Instrumentation berpresisi tinggi untuk profiling inferensi TFLite + Adaptive EMA
 * pada perangkat Android.
 *
 * FITUR:
 * 1. Warm-up: 30 frame pertama dieksklusikan dari metrik evaluasi.
 * 2. Target sampel: minimal 200 frame steady-state.
 * 3. Pre-allocated primitive arrays: Zero-GC selama benchmarking agar tidak mendistorsi latensi.
 * 4. Tidak ada logging per-frame ke Logcat (menghindari I/O bottleneck logcat).
 * 5. Pengukuran waktu berpresisi nanodetik via SystemClock.elapsedRealtimeNanos().
 * 6. Ekspor otomatis ke CSV di app external files directory.
 * 7. Ringkasan statistik (Min, Mean, Median/p50, p95, Max) + Throughput FPS aktual.
 */
public class PerformanceProfiler {

    private static final String TAG = "VeinPerfProfiler";

    public static final int WARMUP_COUNT   = 30;
    public static final int TARGET_SAMPLES = 200;
    private static final int MAX_CAPACITY   = 500;

    private static volatile PerformanceProfiler sInstance = null;

    public static PerformanceProfiler getInstance() {
        if (sInstance == null) {
            synchronized (PerformanceProfiler.class) {
                if (sInstance == null) {
                    sInstance = new PerformanceProfiler();
                }
            }
        }
        return sInstance;
    }

    // Status tracker
    private int mTotalFramesProcessed = 0;
    private int mWarmupFramesCount    = 0;
    private int mSampleCount          = 0;

    private final AtomicInteger mDroppedFramesCount = new AtomicInteger(0);
    private int mDetectedFramesCount = 0;

    // Timestamps untuk throughput FPS (wall-clock)
    private long mBenchmarkStartNs = 0L;
    private long mBenchmarkEndNs   = 0L;
    private boolean mBenchmarkCompleted = false;

    // Pre-allocated arrays untuk data 200+ sampel (satuan: nanodetik)
    private final long[] mPreprocessNs   = new long[MAX_CAPACITY];
    private final long[] mInferenceNs    = new long[MAX_CAPACITY];
    private final long[] mPostprocessNs  = new long[MAX_CAPACITY];
    private final long[] mMaskDecodeNs   = new long[MAX_CAPACITY];
    private final long[] mEmaNs          = new long[MAX_CAPACITY];
    private final long[] mPipelineNs     = new long[MAX_CAPACITY];
    private final boolean[] mHasDetection = new boolean[MAX_CAPACITY];

    // Overlay rendering time (onDraw)
    private final long[] mOnDrawNs       = new long[MAX_CAPACITY];
    private int mOnDrawCount             = 0;

    // ─────────────────────────────────────────────────────────────
    // TEMPORAL PIPELINE SAMPLES (FASE A INSTRUMENTATION)
    // ─────────────────────────────────────────────────────────────
    public static final int MAX_TEMPORAL_SAMPLES = 1500;
    private int mTemporalCount = 0;

    private final long[]   mTpCaptureNs      = new long[MAX_TEMPORAL_SAMPLES];
    private final long[]   mTpInferenceEndNs = new long[MAX_TEMPORAL_SAMPLES];
    private final long[]   mTpDrawNs         = new long[MAX_TEMPORAL_SAMPLES];
    private final double[] mTpPredAgeMs      = new double[MAX_TEMPORAL_SAMPLES];
    private final boolean[] mTpIsFresh       = new boolean[MAX_TEMPORAL_SAMPLES];
    private final boolean[] mTpIsHeld        = new boolean[MAX_TEMPORAL_SAMPLES];
    private final double[] mTpHoldAgeMs      = new double[MAX_TEMPORAL_SAMPLES];
    private final boolean[] mTpIsStale       = new boolean[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpRawConf        = new float[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpRawCx          = new float[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpRawCy          = new float[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpEmaCx          = new float[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpEmaCy          = new float[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpRawL           = new float[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpRawT           = new float[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpRawR           = new float[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpRawB           = new float[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpDispL          = new float[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpDispT          = new float[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpDispR          = new float[MAX_TEMPORAL_SAMPLES];
    private final float[]  mTpDispB          = new float[MAX_TEMPORAL_SAMPLES];
    private final boolean[] mTpHasDet        = new boolean[MAX_TEMPORAL_SAMPLES];

    // ─────────────────────────────────────────────────────────────
    // HYBRID TRACKING TEMPORAL METRICS (PHASE 15 & 16)
    // ─────────────────────────────────────────────────────────────
    public static final int MAX_HYBRID_SAMPLES = 2000;
    private int mHybridCount = 0;

    private final long[]   mHbCaptureNs      = new long[MAX_HYBRID_SAMPLES];
    private final String[] mHbSourceType     = new String[MAX_HYBRID_SAMPLES];
    private final String[] mHbState          = new String[MAX_HYBRID_SAMPLES];
    private final float[]  mHbPredAgeMs      = new float[MAX_HYBRID_SAMPLES];
    private final float[]  mHbRawConf        = new float[MAX_HYBRID_SAMPLES];
    private final float[]  mHbTrackConf      = new float[MAX_HYBRID_SAMPLES];
    private final int[]    mHbPtsInput       = new int[MAX_HYBRID_SAMPLES];
    private final int[]    mHbPtsValid       = new int[MAX_HYBRID_SAMPLES];
    private final float[]  mHbFbErrorPx      = new float[MAX_HYBRID_SAMPLES];
    private final float[]  mHbLkErrorPx      = new float[MAX_HYBRID_SAMPLES];
    private final float[]  mHbDispX          = new float[MAX_HYBRID_SAMPLES];
    private final float[]  mHbDispY          = new float[MAX_HYBRID_SAMPLES];
    private final float[]  mHbCx             = new float[MAX_HYBRID_SAMPLES];
    private final float[]  mHbCy             = new float[MAX_HYBRID_SAMPLES];
    private final float[]  mHbCorrectionDx   = new float[MAX_HYBRID_SAMPLES];
    private final float[]  mHbCorrectionDy   = new float[MAX_HYBRID_SAMPLES];
    private final boolean[] mHbIsReset       = new boolean[MAX_HYBRID_SAMPLES];

    // Tracker aggregate statistics
    private int mTrackerResetCount = 0;
    private int mTrackerCorrectionCount = 0;
    private float mTotalTrackerCorrectionDist = 0f;
    private int mOfStepCount = 0;
    private float mTotalOfDurationMs = 0f;

    // Metadata environment
    private String mDelegateName = "Unknown";
    private String mModelName    = "YoloV8_320.tflite";
    private int mInputSize       = 320;

    private PerformanceProfiler() {
        reset();
    }

    public synchronized void reset() {
        mTotalFramesProcessed = 0;
        mWarmupFramesCount    = 0;
        mSampleCount          = 0;
        mOnDrawCount          = 0;
        mDetectedFramesCount  = 0;
        mDroppedFramesCount.set(0);
        mBenchmarkStartNs     = 0L;
        mBenchmarkEndNs       = 0L;
        mBenchmarkCompleted   = false;
    }

    public void setModelMetadata(String modelName, int inputSize, String delegateName) {
        this.mModelName    = modelName;
        this.mInputSize    = inputSize;
        this.mDelegateName = delegateName;
    }

    /**
     * Catat frame yang di-drop oleh Camera Thread karena AI masih sibuk.
     */
    public void recordDroppedFrame() {
        mDroppedFramesCount.incrementAndGet();
    }

    /**
     * Catat durasi onDraw di OverlayView (rendering cost).
     */
    public synchronized void recordOnDraw(long durationNs) {
        if (mWarmupFramesCount < WARMUP_COUNT) {
            return;
        }
        if (mOnDrawCount < MAX_CAPACITY) {
            mOnDrawNs[mOnDrawCount++] = durationNs;
        }
    }

    /**
     * Catat satu sampel end-to-end temporal pipeline (FASE A).
     * Dipanggil saat OverlayView selesai menggambar deteksi.
     */
    public synchronized void recordTemporalSample(long captureNs,
                                                 long inferenceEndNs,
                                                 long drawNs,
                                                 double predAgeMs,
                                                 boolean isFresh,
                                                 boolean isHeld,
                                                 double holdAgeMs,
                                                 boolean isStale,
                                                 float rawConf,
                                                 float rawCx, float rawCy,
                                                 float emaCx, float emaCy,
                                                 float rawL, float rawT, float rawR, float rawB,
                                                 float dispL, float dispT, float dispR, float dispB,
                                                 boolean hasDet,
                                                 Context context) {
        if (mTemporalCount < MAX_TEMPORAL_SAMPLES) {
            int i = mTemporalCount;
            mTpCaptureNs[i]      = captureNs;
            mTpInferenceEndNs[i] = inferenceEndNs;
            mTpDrawNs[i]         = drawNs;
            mTpPredAgeMs[i]      = predAgeMs;
            mTpIsFresh[i]        = isFresh;
            mTpIsHeld[i]         = isHeld;
            mTpHoldAgeMs[i]      = holdAgeMs;
            mTpIsStale[i]        = isStale;
            mTpRawConf[i]        = rawConf;
            mTpRawCx[i]          = rawCx;
            mTpRawCy[i]          = rawCy;
            mTpEmaCx[i]          = emaCx;
            mTpEmaCy[i]          = emaCy;
            mTpRawL[i]           = rawL;
            mTpRawT[i]           = rawT;
            mTpRawR[i]           = rawR;
            mTpRawB[i]           = rawB;
            mTpDispL[i]          = dispL;
            mTpDispT[i]          = dispT;
            mTpDispR[i]          = dispR;
            mTpDispB[i]          = dispB;
            mTpHasDet[i]         = hasDet;
            mTemporalCount++;

            if (mTemporalCount % 100 == 0 && context != null) {
                exportTemporalCsv(context);
            }
        }
    }

    /**
     * Catat seluruh breakdown waktu untuk satu frame inferensi.
     */
    public synchronized void recordInference(long preprocessNs,
                                            long inferenceNs,
                                            long postprocessNs,
                                            long maskDecodeNs,
                                            long emaNs,
                                            long pipelineNs,
                                            boolean hasDetection,
                                            Context context) {
        mTotalFramesProcessed++;

        // Fase Warm-up (30 frame pertama dieksklusikan)
        if (mWarmupFramesCount < WARMUP_COUNT) {
            mWarmupFramesCount++;
            if (mWarmupFramesCount == WARMUP_COUNT) {
                mBenchmarkStartNs = SystemClock.elapsedRealtimeNanos();
                Log.i(TAG, ">>> WARM-UP SELESAI (" + WARMUP_COUNT + " frames). Memulai pengukuran benchmark...");
            }
            return;
        }

        // Fase Pengukuran (Target 200 sampel)
        if (mSampleCount < MAX_CAPACITY) {
            int idx = mSampleCount;
            mPreprocessNs[idx]  = preprocessNs;
            mInferenceNs[idx]   = inferenceNs;
            mPostprocessNs[idx] = postprocessNs;
            mMaskDecodeNs[idx]  = maskDecodeNs;
            mEmaNs[idx]         = emaNs;
            mPipelineNs[idx]    = pipelineNs;
            mHasDetection[idx]  = hasDetection;

            if (hasDetection) {
                mDetectedFramesCount++;
            }

            mSampleCount++;

            // Progress log setiap 50 frame (ringan, tidak spam logcat)
            if (mSampleCount % 50 == 0) {
                Log.i(TAG, String.format(Locale.US,
                        "Benchmark progress: %d/%d frames (Pipeline: %.2f ms, Model: %.2f ms)",
                        mSampleCount, TARGET_SAMPLES,
                        pipelineNs / 1_000_000.0, inferenceNs / 1_000_000.0));
            }

            // Target 200 sampel tercapai
            if (mSampleCount == TARGET_SAMPLES && !mBenchmarkCompleted) {
                mBenchmarkEndNs = SystemClock.elapsedRealtimeNanos();
                mBenchmarkCompleted = true;
                Log.i(TAG, ">>> TARGET BENCHMARK TERCAPAI (200 frames). Menganalisis statistik...");
                dumpReport(context);
            }
        }
    }

    /**
     * Data class untuk statistik sebuah tahap
     */
    public static class StageStats {
        public String name;
        public int count;
        public double minMs;
        public double meanMs;
        public double medianMs;
        public double p95Ms;
        public double maxMs;

        @Override
        public String toString() {
            return String.format(Locale.US,
                    "%-16s | %5d | %7.2f | %7.2f | %7.2f | %7.2f | %7.2f",
                    name, count, minMs, meanMs, medianMs, p95Ms, maxMs);
        }
    }

    /**
     * Hitung statistik dari array data nanodetik.
     */
    private StageStats computeStats(String stageName, long[] rawNs, int count) {
        StageStats stats = new StageStats();
        stats.name  = stageName;
        stats.count = count;

        if (count == 0) {
            stats.minMs = stats.meanMs = stats.medianMs = stats.p95Ms = stats.maxMs = 0.0;
            return stats;
        }

        double[] msValues = new double[count];
        double sum = 0.0;
        for (int i = 0; i < count; i++) {
            double ms = rawNs[i] / 1_000_000.0;
            msValues[i] = ms;
            sum += ms;
        }

        Arrays.sort(msValues);

        stats.minMs  = msValues[0];
        stats.maxMs  = msValues[count - 1];
        stats.meanMs = sum / count;

        // Median (p50)
        if (count % 2 == 1) {
            stats.medianMs = msValues[count / 2];
        } else {
            stats.medianMs = (msValues[count / 2 - 1] + msValues[count / 2]) / 2.0;
        }

        // p95
        int p95Index = (int) Math.ceil(0.95 * count) - 1;
        if (p95Index < 0) p95Index = 0;
        if (p95Index >= count) p95Index = count - 1;
        stats.p95Ms = msValues[p95Index];

        return stats;
    }

    /**
     * Buat dan cetak laporan lengkap ke Logcat serta ekspor ke file CSV & Markdown.
     */
    public synchronized String dumpReport(Context context) {
        if (mSampleCount == 0) {
            Log.w(TAG, "dumpReport dipanggil tetapi belum ada sampel yang terkumpul.");
            return "No samples collected";
        }

        long endNs = (mBenchmarkEndNs > 0) ? mBenchmarkEndNs : SystemClock.elapsedRealtimeNanos();
        double wallClockSec = (endNs - mBenchmarkStartNs) / 1_000_000_000.0;
        double actualThroughputFps = (wallClockSec > 0) ? (mSampleCount / wallClockSec) : 0.0;

        StageStats sPreprocess  = computeStats("Preprocessing",  mPreprocessNs,  mSampleCount);
        StageStats sInference   = computeStats("TFLite Model",   mInferenceNs,   mSampleCount);
        StageStats sPostprocess = computeStats("Parsing + NMS",  mPostprocessNs, mSampleCount);
        StageStats sMaskDecode  = computeStats("Mask Decoding",  mMaskDecodeNs,  mSampleCount);
        StageStats sEma         = computeStats("Adaptive EMA",   mEmaNs,         mSampleCount);
        StageStats sPipeline    = computeStats("Pipeline Total", mPipelineNs,    mSampleCount);
        StageStats sOnDraw      = computeStats("Overlay onDraw", mOnDrawNs,      mOnDrawCount);

        StringBuilder sb = new StringBuilder();
        sb.append("\n=========================================================================================\n");
        sb.append("                  HASIL PROFILING PERFORMA ANDROID — YOLOv8-seg + EMA                   \n");
        sb.append("=========================================================================================\n");
        sb.append(String.format("Perangkat      : %s %s (Android %s, API %d)\n",
                Build.MANUFACTURER, Build.MODEL, Build.VERSION.RELEASE, Build.VERSION.SDK_INT));
        sb.append(String.format("Hardware       : %s | Platform: %s\n", Build.HARDWARE, Build.BOARD));
        sb.append(String.format("Model TFLite   : %s (%dx%d NCHW)\n", mModelName, mInputSize, mInputSize));
        sb.append(String.format("Runtime/Delegate: %s\n", mDelegateName));
        sb.append(String.format("Warm-up        : %d frames (dieksklusikan)\n", mWarmupFramesCount));
        sb.append(String.format("Sampel Terukur : %d frames (target: %d)\n", mSampleCount, TARGET_SAMPLES));
        sb.append(String.format("Frame Vena Ada : %d / %d (%.1f%%)\n",
                mDetectedFramesCount, mSampleCount, (mDetectedFramesCount * 100.0 / mSampleCount)));
        sb.append(String.format("Frame Terlewat : %d dropped frames (kamera lebih cepat dari AI)\n",
                mDroppedFramesCount.get()));
        sb.append(String.format("Durasi Evaluasi: %.2f detik\n", wallClockSec));
        sb.append(String.format("Throughput AI  : %.2f FPS (aktual wall-clock: frames/durasi)\n", actualThroughputFps));
        sb.append(String.format("Throughput Lat : %.2f FPS (teoretis: 1000 / median_pipeline)\n",
                (sPipeline.medianMs > 0 ? (1000.0 / sPipeline.medianMs) : 0.0)));
        sb.append("-----------------------------------------------------------------------------------------\n");
        sb.append("Tahap Pipeline   | Count |  Min ms |  Mean ms| Med/p50 |  p95 ms |  Max ms\n");
        sb.append("-----------------------------------------------------------------------------------------\n");
        sb.append(sPreprocess.toString()).append("\n");
        sb.append(sInference.toString()).append("\n");
        sb.append(sPostprocess.toString()).append("\n");
        sb.append(sMaskDecode.toString()).append("\n");
        sb.append(sEma.toString()).append("\n");
        sb.append("-----------------------------------------------------------------------------------------\n");
        sb.append(sPipeline.toString()).append("\n");
        sb.append("-----------------------------------------------------------------------------------------\n");
        sb.append(sOnDraw.toString()).append(" (Rendering 60Hz loop)\n");
        sb.append("=========================================================================================\n");

        String reportStr = sb.toString();
        Log.i(TAG, reportStr);

        // Ekspor ke CSV
        exportCsv(context);

        return reportStr;
    }

    /**
     * Ekspor data sampel ke CSV di direktori file eksternal aplikasi.
     */
    private void exportCsv(Context context) {
        if (context == null || mSampleCount == 0) return;
        try {
            File dir = context.getExternalFilesDir(null);
            if (dir == null) dir = context.getFilesDir();

            File csvFile = new File(dir, "benchmark_samples.csv");
            PrintWriter pw = new PrintWriter(new FileWriter(csvFile));
            pw.println("frame_id,preprocess_ms,inference_ms,postprocess_ms,mask_decode_ms,ema_ms,pipeline_ms,has_detection");

            for (int i = 0; i < mSampleCount; i++) {
                pw.println(String.format(Locale.US, "%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%b",
                        i + 1,
                        mPreprocessNs[i] / 1_000_000.0,
                        mInferenceNs[i] / 1_000_000.0,
                        mPostprocessNs[i] / 1_000_000.0,
                        mMaskDecodeNs[i] / 1_000_000.0,
                        mEmaNs[i] / 1_000_000.0,
                        mPipelineNs[i] / 1_000_000.0,
                        mHasDetection[i]));
            }
            pw.flush();
            pw.close();
            Log.i(TAG, "CSV benchmark disimpan di: " + csvFile.getAbsolutePath());

            // Tulis juga ringkasan teks
            File txtFile = new File(dir, "benchmark_summary.txt");
            PrintWriter pwTxt = new PrintWriter(new FileWriter(txtFile));
            pwTxt.print(dumpReportTextOnly());
            pwTxt.flush();
            pwTxt.close();

            // Ekspor juga data temporal jika ada
            exportTemporalCsv(context);
            exportHybridTrackingCsv(context);

        } catch (Exception e) {
            Log.e(TAG, "Gagal menulis file CSV/summary: " + e.getMessage());
        }
    }

    /**
     * Ekspor data temporal pipeline samples ke CSV (FASE A).
     */
    public synchronized void exportTemporalCsv(Context context) {
        if (context == null || mTemporalCount == 0) return;
        try {
            File dir = context.getExternalFilesDir(null);
            if (dir == null) dir = context.getFilesDir();

            File csvFile = new File(dir, "temporal_pipeline_samples.csv");
            PrintWriter pw = new PrintWriter(new FileWriter(csvFile));
            pw.println("frame_id,capture_timestamp_ns,inference_end_ns,draw_timestamp_ns,prediction_age_ms,is_fresh,is_held,hold_age_ms,is_stale,raw_conf,raw_cx,raw_cy,ema_cx,ema_cy,raw_l,raw_t,raw_r,raw_b,disp_l,disp_t,disp_r,disp_b,has_detection");

            for (int i = 0; i < mTemporalCount; i++) {
                pw.println(String.format(Locale.US,
                        "%d,%d,%d,%d,%.2f,%b,%b,%.2f,%b,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%b",
                        i + 1,
                        mTpCaptureNs[i],
                        mTpInferenceEndNs[i],
                        mTpDrawNs[i],
                        mTpPredAgeMs[i],
                        mTpIsFresh[i],
                        mTpIsHeld[i],
                        mTpHoldAgeMs[i],
                        mTpIsStale[i],
                        mTpRawConf[i],
                        mTpRawCx[i], mTpRawCy[i],
                        mTpEmaCx[i], mTpEmaCy[i],
                        mTpRawL[i], mTpRawT[i], mTpRawR[i], mTpRawB[i],
                        mTpDispL[i], mTpDispT[i], mTpDispR[i], mTpDispB[i],
                        mTpHasDet[i]));
            }
            pw.flush();
            pw.close();
            Log.i(TAG, "Temporal CSV (" + mTemporalCount + " rows) disimpan di: " + csvFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e(TAG, "Gagal menulis temporal CSV: " + e.getMessage());
        }
    }

    // ─────────────────────────────────────────────────────────────
    // HYBRID TRACKING METRICS RECORDING & EXPORT (PHASE 15 & 16)
    // ─────────────────────────────────────────────────────────────

    public synchronized void recordOpticalFlowStep(float durationMs, int validPoints, float fbErrorPx) {
        mOfStepCount++;
        mTotalOfDurationMs += durationMs;
    }

    public synchronized void recordTrackerReset() {
        mTrackerResetCount++;
    }

    public synchronized void recordTrackerCorrection(float discrepancyPx) {
        mTrackerCorrectionCount++;
        mTotalTrackerCorrectionDist += discrepancyPx;
    }

    public synchronized void recordHybridTrackingSample(
            long captureTsNs,
            String sourceType,
            String state,
            float predAgeMs,
            float rawConf,
            float trackConf,
            int ptsInput,
            int ptsValid,
            float fbErrorPx,
            float lkErrorPx,
            float dispX,
            float dispY,
            float cx,
            float cy,
            float corrDx,
            float corrDy,
            boolean isReset,
            Context context) {

        if (mHybridCount < MAX_HYBRID_SAMPLES) {
            int idx = mHybridCount;
            mHbCaptureNs[idx]    = captureTsNs;
            mHbSourceType[idx]   = sourceType;
            mHbState[idx]        = state;
            mHbPredAgeMs[idx]    = predAgeMs;
            mHbRawConf[idx]      = rawConf;
            mHbTrackConf[idx]    = trackConf;
            mHbPtsInput[idx]     = ptsInput;
            mHbPtsValid[idx]     = ptsValid;
            mHbFbErrorPx[idx]    = fbErrorPx;
            mHbLkErrorPx[idx]    = lkErrorPx;
            mHbDispX[idx]        = dispX;
            mHbDispY[idx]        = dispY;
            mHbCx[idx]           = cx;
            mHbCy[idx]           = cy;
            mHbCorrectionDx[idx] = corrDx;
            mHbCorrectionDy[idx] = corrDy;
            mHbIsReset[idx]      = isReset;
            mHybridCount++;

            if (mHybridCount % 300 == 0 && context != null) {
                exportHybridTrackingCsv(context);
            }
        }
    }

    public synchronized void exportHybridTrackingCsv(Context context) {
        if (context == null || mHybridCount == 0) return;
        try {
            File dir = context.getExternalFilesDir(null);
            if (dir == null) dir = context.getFilesDir();

            File csvFile = new File(dir, "hybrid_tracking_temporal_metrics.csv");
            PrintWriter pw = new PrintWriter(new FileWriter(csvFile));
            pw.println("frame_id,capture_timestamp_ns,source_type,state,pred_age_ms,raw_conf,track_conf,pts_input,pts_valid,fb_error_px,lk_error_px,disp_x,disp_y,cx,cy,correction_dx,correction_dy,is_reset");

            for (int i = 0; i < mHybridCount; i++) {
                pw.println(String.format(Locale.US,
                        "%d,%d,%s,%s,%.2f,%.4f,%.4f,%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%b",
                        i + 1,
                        mHbCaptureNs[i],
                        mHbSourceType[i],
                        mHbState[i],
                        mHbPredAgeMs[i],
                        mHbRawConf[i],
                        mHbTrackConf[i],
                        mHbPtsInput[i],
                        mHbPtsValid[i],
                        mHbFbErrorPx[i],
                        mHbLkErrorPx[i],
                        mHbDispX[i],
                        mHbDispY[i],
                        mHbCx[i],
                        mHbCy[i],
                        mHbCorrectionDx[i],
                        mHbCorrectionDy[i],
                        mHbIsReset[i]));
            }
            pw.flush();
            pw.close();
            Log.i(TAG, "Hybrid Tracking CSV (" + mHybridCount + " rows) disimpan di: " + csvFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e(TAG, "Gagal menulis hybrid tracking CSV: " + e.getMessage());
        }
    }

    // ─────────────────────────────────────────────────────────────
    // KALMAN TRACKING METRICS RECORDING & EXPORT (PHASE 8 & 16)
    // ─────────────────────────────────────────────────────────────
    public static final int MAX_KALMAN_SAMPLES = 2500;
    private int mKalmanCount = 0;

    private final long[]    mKmCaptureNs       = new long[MAX_KALMAN_SAMPLES];
    private final String[]  mKmState           = new String[MAX_KALMAN_SAMPLES];
    private final float[]   mKmPredAgeMs       = new float[MAX_KALMAN_SAMPLES];
    private final float[]   mKmConfidence      = new float[MAX_KALMAN_SAMPLES];
    private final float[]   mKmCx              = new float[MAX_KALMAN_SAMPLES];
    private final float[]   mKmCy              = new float[MAX_KALMAN_SAMPLES];
    private final float[]   mKmWidth           = new float[MAX_KALMAN_SAMPLES];
    private final float[]   mKmHeight          = new float[MAX_KALMAN_SAMPLES];
    private final boolean[] mKmAccepted        = new boolean[MAX_KALMAN_SAMPLES];
    private final float[]   mKmDiscrepancyPx   = new float[MAX_KALMAN_SAMPLES];
    private final boolean[] mKmIsReset         = new boolean[MAX_KALMAN_SAMPLES];

    private int   mKmMeasurementCount    = 0;
    private int   mKmAcceptedCount       = 0;
    private float mTotalKmDiscrepancyPx  = 0f;

    public synchronized void recordKalmanMeasurement(boolean accepted, float discrepancyPx, boolean isReset) {
        mKmMeasurementCount++;
        if (accepted) mKmAcceptedCount++;
        mTotalKmDiscrepancyPx += discrepancyPx;
    }

    public synchronized void recordKalmanSample(
            long captureTsNs,
            String state,
            float predAgeMs,
            float confidence,
            float cx,
            float cy,
            float width,
            float height,
            boolean accepted,
            float discrepancyPx,
            boolean isReset,
            Context context) {

        if (mKalmanCount < MAX_KALMAN_SAMPLES) {
            int idx = mKalmanCount;
            mKmCaptureNs[idx]     = captureTsNs;
            mKmState[idx]         = state;
            mKmPredAgeMs[idx]     = predAgeMs;
            mKmConfidence[idx]    = confidence;
            mKmCx[idx]            = cx;
            mKmCy[idx]            = cy;
            mKmWidth[idx]         = width;
            mKmHeight[idx]        = height;
            mKmAccepted[idx]      = accepted;
            mKmDiscrepancyPx[idx] = discrepancyPx;
            mKmIsReset[idx]       = isReset;
            mKalmanCount++;

            if (mKalmanCount % 300 == 0 && context != null) {
                exportKalmanCsv(context);
            }
        }
    }

    public synchronized void exportKalmanCsv(Context context) {
        if (context == null || mKalmanCount == 0) return;
        try {
            File dir = context.getExternalFilesDir(null);
            if (dir == null) dir = context.getFilesDir();

            File csvFile = new File(dir, "kalman_temporal_metrics.csv");
            PrintWriter pw = new PrintWriter(new FileWriter(csvFile));
            pw.println("frame_id,capture_timestamp_ns,state,pred_age_ms,confidence,cx,cy,width,height,accepted,discrepancy_px,is_reset");

            for (int i = 0; i < mKalmanCount; i++) {
                pw.println(String.format(Locale.US,
                        "%d,%d,%s,%.2f,%.4f,%.2f,%.2f,%.2f,%.2f,%b,%.2f,%b",
                        i + 1,
                        mKmCaptureNs[i],
                        mKmState[i],
                        mKmPredAgeMs[i],
                        mKmConfidence[i],
                        mKmCx[i],
                        mKmCy[i],
                        mKmWidth[i],
                        mKmHeight[i],
                        mKmAccepted[i],
                        mKmDiscrepancyPx[i],
                        mKmIsReset[i]));
            }
            pw.flush();
            pw.close();
            Log.i(TAG, "Kalman CSV (" + mKalmanCount + " rows) disimpan di: " + csvFile.getAbsolutePath());
        } catch (Exception e) {
            Log.e(TAG, "Gagal menulis Kalman CSV: " + e.getMessage());
        }
    }

    private String dumpReportTextOnly() {
        long endNs = (mBenchmarkEndNs > 0) ? mBenchmarkEndNs : SystemClock.elapsedRealtimeNanos();
        double wallClockSec = (endNs - mBenchmarkStartNs) / 1_000_000_000.0;
        double actualThroughputFps = (wallClockSec > 0) ? (mSampleCount / wallClockSec) : 0.0;

        StageStats sPreprocess  = computeStats("Preprocessing",  mPreprocessNs,  mSampleCount);
        StageStats sInference   = computeStats("TFLite Model",   mInferenceNs,   mSampleCount);
        StageStats sPostprocess = computeStats("Parsing + NMS",  mPostprocessNs, mSampleCount);
        StageStats sMaskDecode  = computeStats("Mask Decoding",  mMaskDecodeNs,  mSampleCount);
        StageStats sEma         = computeStats("Adaptive EMA",   mEmaNs,         mSampleCount);
        StageStats sPipeline    = computeStats("Pipeline Total", mPipelineNs,    mSampleCount);
        StageStats sOnDraw      = computeStats("Overlay onDraw", mOnDrawNs,      mOnDrawCount);

        StringBuilder sb = new StringBuilder();
        sb.append("Device: ").append(Build.MANUFACTURER).append(" ").append(Build.MODEL)
          .append(" (Android ").append(Build.VERSION.RELEASE).append(", API ").append(Build.VERSION.SDK_INT).append(")\n");
        sb.append("Model: ").append(mModelName).append(" | Delegate: ").append(mDelegateName).append("\n");
        sb.append("Warmup: ").append(mWarmupFramesCount).append(" frames | Measured: ").append(mSampleCount).append(" frames\n");
        sb.append("Wall-clock duration: ").append(String.format(Locale.US, "%.2f s", wallClockSec)).append("\n");
        sb.append("Throughput actual: ").append(String.format(Locale.US, "%.2f FPS", actualThroughputFps)).append("\n");
        sb.append("Stage,Min_ms,Mean_ms,Median_ms,p95_ms,Max_ms\n");
        sb.append(formatStatsRow(sPreprocess)).append("\n");
        sb.append(formatStatsRow(sInference)).append("\n");
        sb.append(formatStatsRow(sPostprocess)).append("\n");
        sb.append(formatStatsRow(sMaskDecode)).append("\n");
        sb.append(formatStatsRow(sEma)).append("\n");
        sb.append(formatStatsRow(sPipeline)).append("\n");
        sb.append(formatStatsRow(sOnDraw)).append("\n");
        return sb.toString();
    }

    private String formatStatsRow(StageStats s) {
        return String.format(Locale.US, "%s,%.2f,%.2f,%.2f,%.2f,%.2f",
                s.name, s.minMs, s.meanMs, s.medianMs, s.p95Ms, s.maxMs);
    }

    public int getSampleCount() {
        return mSampleCount;
    }

    public boolean isBenchmarkCompleted() {
        return mBenchmarkCompleted;
    }

    public String getDelegateName() {
        return mDelegateName;
    }
}

