# Accuracy-first pipeline v2

Date: 2026-11-10 (Asia/Bangkok)

## Outcome

The work stopped at Gate 1, as required by the task. The current MediaProjection
source captures this application's own overlay window on the physical SM-T295.
`FLAG_SECURE` was tested and rejected because it makes the full inference frame
black whenever the overlay is visible.

See `reports/model_input_feedback_audit.md` and the three
`reports/model_input_*.png` artifacts for measured evidence.

## Phase status

| Gate / phase | Status | Evidence or reason |
|---|---|---|
| Gate 1 — self-overlay feedback | **FAIL** | HUD is present in an exact pre-inference MediaProjection frame |
| Gate 1 — safe source separation | **BLOCKED** | Vendor app exposes no supported raw-frame component; `FLAG_SECURE` blanks the source |
| Gate 2 — hard-negative FP32 export | NOT RUN | Gate 1 has not passed |
| RAW old-vs-new Android A/B | NOT RUN | Candidate was not exported/deployed; no claims made |
| Visual state / state machine | NOT RUN | Temporal tuning prohibited while Gate 1 fails |
| Mask-quality analysis | NOT RUN | No Gate-1-clean, labelled Android output set |
| Dynamic coast / predictor / One Euro | NOT RUN | No Gate-1-clean timing and ground truth |
| Long-video validation | NOT RUN | No long recording is present in the workspace and Gate 1 fails |
| Optional verifier / INT8 | NOT RUN | Accuracy gates have not passed |

## Model integrity

- `app/src/main/assets/YoloV8_320.tflite` was not overwritten.
- `runs/segment/runs/yolov8_clean_hardneg/weights/best.pt` was located.
- No file named `YoloV8_320_clean_hardneg_fp32.tflite` was created or deployed,
  because doing so before Gate 1 passes would violate the required sequence.

The existing hard-negative report also records 27/61 false-positive frames for
the retrained checkpoint versus 21/61 for the baseline at confidence ≥ 0.40.
That result does not justify deployment without the requested RAW Android A/B.

## Code change retained

`VeinDetectionService` now supports one-shot exact pre-inference frame capture
for `no_overlay`, `with_overlay`, and `with_hud`. Inference is no longer disabled
merely because the presentation view is hidden, allowing the no-overlay audit
to exercise the same detector path.

No production temporal behavior, thresholds, model asset, or decoder was
changed in this gate-limited pass.

## Required next step

Acquire a supported raw-frame interface from SonoStar (SDK, documented stream,
or exported frame service) and move YOLO input before display composition. Then
repeat Gate 1 on the SM-T295. Only a passing run authorizes Gate 2.

## Production recommendation

**Do not nominate a production temporal configuration yet.** The authoritative
semantic detector is currently fed by a source with a demonstrated application
feedback path. Recommending RAW, EMA, LK, Kalman, alpha-beta, or One Euro before
source separation would not be evidence-based.
