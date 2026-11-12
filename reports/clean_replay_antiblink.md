# Clean Video Replay + RAW YOLO Continuity Audit

## Result

The clean replay gate passed and the selected production path is:

`MP4 -> RecordedVideoFrameSource -> undecorated Bitmap -> YOLO -> presentation copy -> overlay`

The official replay source was
`Data_Citra_USG/Subjek_1/WirelessUSG2026-07-29-16-12-25.mp4`
(199 frames, 1080 x 632, 24 fps, 8.29 s). The initially discovered
`VideoUSG.mp4` was rejected before A/B because it already had an old `Mode: KALMAN`
HUD burned into the recording. It was not used for the reported measurements.

The device was a physical Samsung SM-T295 running Android 11. Both models ran with
4 CPU threads, XNNPACK disabled, the same confidence threshold (0.35), NMS (0.45),
ROI, letterbox, NCHW normalization, decoded frame indices, and device.

## Clean source and zero-feedback proof

`FrameSource` now defines ownership of raw bitmaps. It has two concrete Android-native
implementations:

- `RecordedVideoFrameSource`: uses `MediaMetadataRetriever.getFrameAtIndex()` and a fixed
  source-frame sequence. It never reads display pixels.
- `MediaProjectionFrameSource`: encapsulates the legacy/live screen-capture source behind
  the same boundary.

`ReplayActivity` constructs only `RecordedVideoFrameSource`; it does not request or construct
MediaProjection. The raw decoder bitmap is hashed and passed to `YoloDetector.detect()` before
a separate bitmap copy is posted to `ReplayDisplayView`.

The exact pre-inference samples are:

- `replay_model_input_01.png`: before any detection was presented.
- `replay_model_input_02.png`: while bbox and mask were visible in the presentation layer.
- `replay_model_input_03.png`: while bbox, mask, and compact HUD were visible.

All three contain only source-video pixels: no application bbox, segmentation color,
confidence/depth label, or debug HUD. The context and frame indices are recorded in
`replay_model_input_context.csv`.

## RAW YOLO continuity

RAW means the highest-confidence post-NMS model result with no EMA, LK, Kalman, confidence
hysteresis, spatial gate, prediction, hold, or smoothing. All 199 source frames completed
inference and are recorded in `raw_yolo_video_detections.csv`.

| Metric | Baseline FP32 |
|---|---:|
| Completed fresh inferences | 199 |
| Detection-present frames | 117 |
| Overall presence rate | 58.79% |
| Presence while reference target is valid | 111 / 111 (100.00%) |
| False-positive frames while reference target is absent | 6 / 88 |
| False blink events while target remains valid | 0 |
| Detection streaks | 38, 38, 2, 39 |
| Miss streaks | 50, 30, 2 |
| Longest consecutive RAW miss | 50 inferences / 2.083 s of source video |
| Mean fresh-inference interval | 355.88 ms |
| Median fresh-inference interval | 343.68 ms |
| Mean interval between detection-present completions | 607.45 ms |
| Mean end-to-end inference pipeline latency | 324.96 ms |

The 50-frame and 30-frame gaps coincide with reference intervals where the vein target is
invalid/absent; they are not false detector blink. Relative to the existing project
morphology-assisted reference (30 px match gate), the clean baseline has zero false misses.

`raw_yolo_replay_output.mp4` is deliberately unsmoothed: a bbox is drawn only for a fresh
RAW detection on that source frame.

## Old model versus hard-negative FP32

The candidate checkpoint was exported separately as
`YoloV8_320_clean_hardneg_fp32.tflite`. It is true FP32 with input
`[1,3,320,320]` and outputs `[1,37,2100]` plus `[1,32,80,80]`. The old model was not
overwritten.

All 199 raw input hashes match between A and B (`0` mismatches).

| Metric | `YoloV8_320.tflite` | Hard-negative FP32 |
|---|---:|---:|
| Presence | 117 / 199 (58.79%) | 159 / 199 (79.90%) |
| True positives | 111 | 111 |
| False positives | **6** | 48 |
| False negatives | 0 | 0 |
| Precision on replay reference | **94.87%** | 69.81% |
| Recall on replay reference | 100.00% | 100.00% |
| Mean matched centroid error | **0.71 px** | 2.75 px |
| Mean confidence when present | **0.832** | 0.725 |
| Mean mask foreground ratio | 0.750 | 0.766 |
| Mean pipeline latency | **324.96 ms** | 350.75 ms |

Every detection from both models had a decoded mask. This replay has center/validity reference,
not per-pixel mask ground truth, so foreground ratio is reported but is not mislabeled as mask
IoU. Operationally, the hard-negative model is worse: it draws 48 masks while the reference
target is absent, versus 6 for the baseline. The baseline model is therefore selected.

Per-frame values are in `raw_model_ab_comparison.csv`; diagnostic videos are
`raw_yolo_replay_output_old.mp4` and `raw_yolo_replay_output_hardneg.mp4`.

## Minimal anti-blink state machine

The selected mode uses only `SEARCH`, `LOCKED`, `COAST`, and `LOST`:

- A fresh detection enters `LOCKED` and remains authoritative.
- The first missed completed inference always enters `COAST`; it cannot blank the overlay.
- A fresh result in `COAST` returns immediately to `LOCKED`.
- A subsequent miss beyond the measured time window enters `LOST`.
- Only `LOST` removes bbox and mask. `SEARCH` has no target to display.

LOCKED and COAST use exactly the same bbox color, mask color, line width, and opacity. There is
no blink, pulse, or fade. State color is permitted only in the compact debug HUD.

### Coast window trial

| Window | Blink count | Max hidden target | Max ghost | Total ghost | Mean reacquisition |
|---:|---:|---:|---:|---:|---:|
| 1.0 x = 343.68 ms | **0** | 0 ms | **1019.52 ms** | **3061.80 ms** | 0 ms |
| 1.5 x = 515.51 ms | **0** | 0 ms | **1019.52 ms** | **3061.80 ms** | 0 ms |
| 2.0 x = 687.35 ms | 0 | 0 ms | 1367.62 ms | 4157.59 ms | 0 ms |

The selected value is 1.0 x median, rounded to **344 ms** in code. It has the same zero-blink
result as 1.5 x but no larger time budget, and it avoids the extra ghost duration measured at
2.0 x. The measured maximum ghost remains below a multi-second hold; it includes the baseline
model's short false-positive runs plus the mandatory one-update COAST.

## Visual-only display interpolation

The bbox is interpolated only in `OverlayView`; the state machine and YOLO coordinates are not
modified. The fresh mask is authoritative and is translated/scaled with the displayed bbox
during the short transition. A centroid correction above 110 px snaps immediately.

| Duration | Mean 60 Hz display step | Steps > 10 px | Max visual lag | Detector changed |
|---:|---:|---:|---:|:---:|
| none | 5.187 px | 6 | 0 px | no |
| 50 ms | 1.749 px | 7 | 40.32 px | no |
| **75 ms** | **1.052 px** | **5** | 48.77 px | no |
| 100 ms | 0.877 px | 6 | 50.40 px | no |

The selected 75 ms transition reduces the mean display step by 79.7% and gives the fewest
steps above 10 px. The maximum step remains 169.31 px by design because large corrections reset
immediately instead of animating through a wrong location.

## Required answers

1. **Was previous blinking caused partly by self-overlay feedback?** Yes. Gate 1 proved the
   exact MediaProjection bitmap contained the app overlay. With direct clean replay, the
   selected RAW detector has zero false blink while the reference target remains valid.
2. **RAW YOLO presence rate after feedback removal?** 117/199 = **58.79% overall**, and
   **100.00% on the 111 target-valid frames**.
3. **Longest consecutive RAW miss?** **50 completed inferences** (2.083 s of source video),
   entirely inside a target-invalid interval; longest false miss while target is valid is zero.
4. **Which FP32 model is better?** **`YoloV8_320.tflite` (old model)**. The hard-negative model
   preserves recall but increases replay false positives from 6 to 48 and is slower.
5. **Best coast duration?** **344 ms (1.0 x measured median fresh-inference interval)**.
6. **Visible blink events after anti-blink?** **0** on this replay/reference.
7. **Does short interpolation help without changing accuracy?** Yes. **75 ms** reduces mean
   visual step by 79.7%; detector coordinates and accuracy are unchanged.

No INT8 export, retraining, Optical Flow, or Kalman tuning was performed.
