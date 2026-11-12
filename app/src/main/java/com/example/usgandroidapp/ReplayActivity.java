package com.example.usgandroidapp;

import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.view.WindowManager;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

/**
 * Device-side deterministic RAW replay audit.
 *
 * <p>Launch with adb extras: replay_path, model_asset, run_label, frame_step. The raw MP4
 * decoder bitmap is hashed, optionally saved, and passed to YOLO before a copy reaches the
 * presentation view. MediaProjection is not requested or constructed in this Activity.</p>
 */
public final class ReplayActivity extends Activity {
    private static final String TAG = "ReplayAudit";
    private static final float ROI_WIDTH_RATIO = 0.60f;
    private static final float ROI_HEIGHT_RATIO = 0.45f;
    private static final float ROI_TOP_RATIO = 0.07f;

    private ReplayDisplayView displayView;
    private RecordedVideoFrameSource frameSource;
    private YoloDetector detector;
    private PrintWriter csv;
    private File outputDir;
    private String runLabel;
    private String modelAsset;
    private int completedCount;
    private int detectionCount;
    private long replayStartNs;
    private long lastCompletionNs;
    private boolean previousHadDetection;
    private boolean audit01Saved;
    private boolean audit02Saved;
    private boolean audit03Saved;
    private volatile boolean finished;

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        displayView = new ReplayDisplayView(this);
        setContentView(displayView);

        String replayPath = getIntent().getStringExtra("replay_path");
        modelAsset = getIntent().getStringExtra("model_asset");
        runLabel = getIntent().getStringExtra("run_label");
        int frameStep = getIntent().getIntExtra("frame_step", 10);
        if (modelAsset == null || modelAsset.trim().isEmpty()) {
            modelAsset = YoloDetector.DEFAULT_MODEL_PATH;
        }
        if (runLabel == null || runLabel.trim().isEmpty()) runLabel = "raw_replay";

        try {
            if (replayPath == null || !new File(replayPath).isFile()) {
                throw new IllegalArgumentException("Replay file not found: " + replayPath);
            }
            File root = new File(getExternalFilesDir(null), "replay_results");
            outputDir = new File(root, safeName(runLabel));
            if (!outputDir.exists() && !outputDir.mkdirs()) {
                throw new IllegalStateException("Cannot create " + outputDir);
            }
            csv = new PrintWriter(new FileWriter(new File(outputDir,
                    "raw_yolo_video_detections.csv"), false));
            csv.println("run_label,model,frame_index,video_timestamp_ms,frame_sha256,"
                    + "completion_elapsed_ms,fresh_inference_interval_ms,"
                    + "detection_present,confidence,left,top,right,bottom,mask_present,"
                    + "mask_foreground_ratio,preprocess_ms,inference_ms,postprocess_ms,"
                    + "mask_decode_ms,total_latency_ms");

            YoloDetector.RuntimeConfig config = new YoloDetector.RuntimeConfig();
            config.modelPath = modelAsset;
            config.numThreads = 4;
            config.useXnnpack = false;
            config.enableGpu = false;
            config.enableNnapi = false;
            detector = new YoloDetector(this, config);
            frameSource = new RecordedVideoFrameSource(replayPath, frameStep);
            replayStartNs = SystemClock.elapsedRealtimeNanos();
            lastCompletionNs = 0L;
            Log.i(TAG, "START label=" + runLabel + " model=" + modelAsset
                    + " source=" + frameSource.getName());
            frameSource.start(sourceListener);
        } catch (Throwable error) {
            fail(error);
        }
    }

    private final FrameSource.Listener sourceListener = new FrameSource.Listener() {
        @Override
        public void onFrame(FrameSource.Frame sourceFrame) {
            processFrame(sourceFrame);
        }

        @Override
        public void onCompleted() {
            complete();
        }

        @Override
        public void onError(Throwable error) {
            fail(error);
        }
    };

    private void processFrame(FrameSource.Frame sourceFrame) {
        Bitmap raw = sourceFrame.bitmap;
        try {
            String hash = sha256(raw);
            boolean overlayVisibleBeforeInference = previousHadDetection;
            boolean hudVisibleBeforeInference = audit02Saved;

            if (!audit01Saved) {
                saveAudit(raw, "replay_model_input_01.png", sourceFrame,
                        false, false, "before any detection");
                audit01Saved = true;
            } else if (!audit02Saved && overlayVisibleBeforeInference) {
                saveAudit(raw, "replay_model_input_02.png", sourceFrame,
                        true, false, "bbox and mask visible in presentation layer");
                audit02Saved = true;
            } else if (!audit03Saved && overlayVisibleBeforeInference && audit02Saved) {
                saveAudit(raw, "replay_model_input_03.png", sourceFrame,
                        true, true, "bbox, mask, and HUD visible in presentation layer");
                audit03Saved = true;
            }

            int width = raw.getWidth();
            int height = raw.getHeight();
            int roiW = Math.max(1, (int) (width * ROI_WIDTH_RATIO));
            int roiH = Math.max(1, (int) (height * ROI_HEIGHT_RATIO));
            int roiL = (width - roiW) / 2;
            int roiT = (int) (height * ROI_TOP_RATIO);

            YoloDetector.StageTimings timings = new YoloDetector.StageTimings();
            long totalStart = SystemClock.elapsedRealtimeNanos();
            List<YoloDetector.Detection> results = detector.detect(raw, roiL, roiT,
                    roiW, roiH, timings, sourceFrame.captureTimestampNs);
            long totalNs = SystemClock.elapsedRealtimeNanos() - totalStart;
            long completionNs = SystemClock.elapsedRealtimeNanos();
            double completionElapsedMs = (completionNs - replayStartNs) / 1_000_000.0;
            double freshIntervalMs = lastCompletionNs == 0L
                    ? 0.0 : (completionNs - lastCompletionNs) / 1_000_000.0;
            lastCompletionNs = completionNs;

            // RAW means exactly the highest-confidence post-NMS detector result. There is no
            // temporal threshold, spatial gate, smoothing, prediction, or grace hold here.
            YoloDetector.Detection best = highestConfidence(results);
            recycleUnselectedMasks(results, best);
            boolean present = best != null;
            completedCount++;
            if (present) detectionCount++;

            writeCsv(sourceFrame, hash, best, timings, totalNs,
                    completionElapsedMs, freshIntervalMs);
            previousHadDetection = present;

            Bitmap displayFrame = raw.copy(Bitmap.Config.ARGB_8888, false);
            YoloDetector.Detection displayDetection = copyForDisplay(best);
            boolean hud = audit02Saved;
            runOnUiThread(() -> displayView.setPresentation(displayFrame, displayDetection,
                    hud, sourceFrame.frameIndex, sourceFrame.mediaTimestampUs));

            if (best != null && best.maskBitmap != null && !best.maskBitmap.isRecycled()) {
                best.maskBitmap.recycle();
            }
            if (completedCount % 25 == 0) {
                csv.flush();
                Log.i(TAG, "PROGRESS label=" + runLabel + " completed=" + completedCount
                        + " detections=" + detectionCount);
            }
        } catch (Throwable error) {
            fail(error);
        } finally {
            if (raw != null && !raw.isRecycled()) raw.recycle();
        }
    }

    private static YoloDetector.Detection highestConfidence(List<YoloDetector.Detection> list) {
        if (list == null || list.isEmpty()) return null;
        return Collections.max(list, Comparator.comparingDouble(d -> d.confidence));
    }

    private static void recycleUnselectedMasks(List<YoloDetector.Detection> list,
                                               YoloDetector.Detection selected) {
        if (list == null) return;
        for (YoloDetector.Detection item : list) {
            if (item != selected && item.maskBitmap != null && !item.maskBitmap.isRecycled()) {
                item.maskBitmap.recycle();
            }
        }
    }

    private void writeCsv(FrameSource.Frame frame, String hash, YoloDetector.Detection det,
                          YoloDetector.StageTimings t, long totalNs,
                          double completionElapsedMs, double freshIntervalMs) {
        boolean present = det != null;
        csv.printf(Locale.US,
                "%s,%s,%d,%.3f,%s,%.3f,%.3f,%s,%.6f,%.3f,%.3f,%.3f,%.3f,%s,%.6f,"
                        + "%.3f,%.3f,%.3f,%.3f,%.3f%n",
                runLabel, modelAsset, frame.frameIndex, frame.mediaTimestampUs / 1000.0,
                hash, completionElapsedMs, freshIntervalMs,
                Boolean.toString(present), present ? det.confidence : 0f,
                present ? det.left : 0f, present ? det.top : 0f,
                present ? det.right : 0f, present ? det.bottom : 0f,
                Boolean.toString(present && det.maskBitmap != null), maskRatio(det),
                nsToMs(t.preprocessNs), nsToMs(t.inferenceNs), nsToMs(t.postprocessNs),
                nsToMs(t.maskDecodeNs), nsToMs(totalNs));
    }

    private static double maskRatio(YoloDetector.Detection det) {
        if (det == null || det.maskBitmap == null || det.maskBitmap.isRecycled()) return 0.0;
        Bitmap mask = det.maskBitmap;
        int[] pixels = new int[mask.getWidth() * mask.getHeight()];
        mask.getPixels(pixels, 0, mask.getWidth(), 0, 0, mask.getWidth(), mask.getHeight());
        int foreground = 0;
        for (int pixel : pixels) if ((pixel >>> 24) != 0) foreground++;
        return pixels.length == 0 ? 0.0 : (double) foreground / pixels.length;
    }

    private static YoloDetector.Detection copyForDisplay(YoloDetector.Detection src) {
        if (src == null) return null;
        YoloDetector.Detection dst = new YoloDetector.Detection();
        dst.left = src.left; dst.top = src.top; dst.right = src.right; dst.bottom = src.bottom;
        dst.cx = src.cx; dst.cy = src.cy; dst.confidence = src.confidence;
        if (src.maskBitmap != null && !src.maskBitmap.isRecycled()) {
            dst.maskBitmap = src.maskBitmap.copy(Bitmap.Config.ARGB_8888, false);
        }
        return dst;
    }

    private void saveAudit(Bitmap raw, String fileName, FrameSource.Frame frame,
                           boolean overlayVisible, boolean hudVisible, String context)
            throws Exception {
        File output = new File(outputDir, fileName);
        try (FileOutputStream stream = new FileOutputStream(output)) {
            if (!raw.compress(Bitmap.CompressFormat.PNG, 100, stream)) {
                throw new IllegalStateException("PNG encode failed for " + output);
            }
        }
        File contextFile = new File(outputDir, "replay_model_input_context.csv");
        boolean writeHeader = !contextFile.exists();
        try (PrintWriter out = new PrintWriter(new FileWriter(contextFile, true))) {
            if (writeHeader) {
                out.println("file,frame_index,video_timestamp_ms,overlay_visible,hud_visible,context");
            }
            out.printf(Locale.US, "%s,%d,%.3f,%s,%s,%s%n", fileName, frame.frameIndex,
                    frame.mediaTimestampUs / 1000.0, overlayVisible, hudVisible, context);
        }
        Log.i(TAG, "AUDIT_SAVED " + output.getAbsolutePath());
    }

    private static String sha256(Bitmap bitmap) throws Exception {
        ByteBuffer pixels = ByteBuffer.allocate(bitmap.getRowBytes() * bitmap.getHeight());
        bitmap.copyPixelsToBuffer(pixels);
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] value = digest.digest(pixels.array());
        StringBuilder result = new StringBuilder(value.length * 2);
        for (byte b : value) result.append(String.format(Locale.US, "%02x", b & 0xff));
        return result.toString();
    }

    private synchronized void complete() {
        if (finished) return;
        finished = true;
        closeResources(false);
        try {
            JSONObject summary = new JSONObject();
            summary.put("status", "complete");
            summary.put("run_label", runLabel);
            summary.put("model", modelAsset);
            summary.put("frame_source", "RecordedVideoFrameSource");
            summary.put("media_projection_used", false);
            summary.put("completed_inferences", completedCount);
            summary.put("detections", detectionCount);
            summary.put("audit_samples", (audit01Saved ? 1 : 0) + (audit02Saved ? 1 : 0)
                    + (audit03Saved ? 1 : 0));
            try (PrintWriter out = new PrintWriter(new FileWriter(
                    new File(outputDir, "run_summary.json"), false))) {
                out.println(summary.toString(2));
            }
        } catch (Exception error) {
            Log.e(TAG, "Cannot save summary", error);
        }
        Log.i(TAG, "REPLAY_AUDIT_COMPLETE label=" + runLabel + " output=" + outputDir);
        runOnUiThread(() -> displayView.postDelayed(this::finish, 800L));
    }

    private synchronized void fail(Throwable error) {
        if (finished) return;
        finished = true;
        Log.e(TAG, "REPLAY_AUDIT_FAILED label=" + runLabel, error);
        if (outputDir != null) {
            try (PrintWriter out = new PrintWriter(new FileWriter(
                    new File(outputDir, "run_error.txt"), false))) {
                error.printStackTrace(out);
            } catch (Exception ignored) {}
        }
        closeResources(true);
        runOnUiThread(this::finish);
    }

    private void closeResources(boolean stopSource) {
        if (stopSource && frameSource != null) frameSource.stop();
        if (csv != null) { csv.flush(); csv.close(); csv = null; }
        if (detector != null) { detector.close(); detector = null; }
    }

    @Override
    protected void onDestroy() {
        if (frameSource != null) frameSource.stop();
        closeResources(false);
        if (displayView != null) displayView.release();
        super.onDestroy();
    }

    private static double nsToMs(long ns) { return ns / 1_000_000.0; }

    private static String safeName(String value) {
        return value.replaceAll("[^A-Za-z0-9._-]", "_");
    }
}
