# Model-input feedback audit

Date: 2026-11-10 (Asia/Bangkok)  
Target device: Samsung SM-T295, Android 11  
App: `com.example.usgandroidapp`

## Gate result

**FAIL — temporal tuning and model A/B work must remain stopped.**

The inference source is a full-display MediaProjection virtual display using
`VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR`. The presentation is a full-screen
`TYPE_APPLICATION_OVERLAY` window. On the physical SM-T295, pixels drawn by
that window were observed in the exact bitmap delivered by `ImageReader` and
passed to `YoloDetector.detect(...)`.

This is a capture → inference → overlay → capture feedback path. A target box,
mask, or label drawn over the USG ROI can become content in a later detector
input. Temporal tuning cannot make that architecture safe.

## Exact input-frame evidence

The service now has a one-shot audit action that saves the same `Bitmap`
object immediately before ROI crop, letterbox, NCHW encoding, and TFLite
inference. It does not use an Android screenshot API and does not capture the
view after rendering.

Files exported from the SM-T295:

- `reports/model_input_no_overlay.png`
- `reports/model_input_with_overlay.png`
- `reports/model_input_with_hud.png`

All three are 1280 × 800 PNG frames read from MediaProjection. The controlled
source places a benchmark USG image in the configured detector ROI.

Observed results:

| Condition | SHA-256 | Observation |
|---|---|---|
| overlay disabled | `D12067459CF6B66034A17625BFBC77E19498527C0A742D5837CB326AA8996A02` | Raw display source; no app overlay |
| overlay enabled, HUD disabled | `D12067459CF6B66034A17625BFBC77E19498527C0A742D5837CB326AA8996A02` | No target was accepted during this controlled capture, so no target graphic was present |
| overlay enabled, HUD enabled | `6ACF68384578CF155A2160A5EA3DE7D070724E1904A8EFBE7C3BB176881F901C` | The app's `LOST / mode:KALMAN` HUD is visibly present in the pre-inference frame |

The HUD capture proves that the overlay window is part of the MediaProjection
source. The HUD happens to be outside the current YOLO ROI in this test. Target
boxes and masks are drawn by the same captured window at detector coordinates,
which overlap the YOLO ROI whenever a target exists. The identical
`with_overlay` frame must not be interpreted as proof that target graphics are
excluded; it only records that the state was LOST for that capture.

## Rejected mitigation: `FLAG_SECURE`

`WindowManager.LayoutParams.FLAG_SECURE` was tested on the physical SM-T295.
It does exclude the overlay pixels, but Samsung's Android 11 compositor blanks
the entire full-screen secure window in the MediaProjection output.

Measured result while the source image remained visible on the tablet:

- overlay disabled: valid 1280 × 800 USG frame, 140,418-byte PNG;
- overlay enabled: all-black 1280 × 800 frame, 5,916-byte PNG;
- overlay + HUD enabled: the same all-black frame, 5,916-byte PNG.

This mitigation was rejected and removed. An all-black detector input does not
pass Gate 1.

## Source-separation investigation

The installed vendor app is WirelessUSG 3.6.56
(`com.sonostar.wirelessusg`). APK/manifest inspection found:

- probe/network code referencing `192.168.1.1` and TCP/UDP transports;
- an internal `USImageView` whose private bitmap is updated before drawing;
- only `MainActivity` is exported;
- its `FileProvider` is not exported;
- no exported service/provider or documented callback for live raw frames.

Therefore this project cannot safely read the vendor app's pre-overlay bitmap
through a supported Android component as currently installed.

## Safest required architecture

Use a raw-frame producer, not MediaProjection, for inference:

1. Obtain the vendor's supported Android SDK or frame callback for the probe.
2. Decode the probe stream into an immutable raw USG bitmap/buffer owned by
   this app (or receive it through a vendor-provided bound service/AIDL/shared
   memory interface).
3. Pass that raw buffer to YOLO.
4. Render the USG frame and this app's overlay separately for display only.
5. Do not feed the composed display back into YOLO.

Reverse-engineering and reimplementing the proprietary probe protocol is not a
safe production shortcut for a medical imaging pipeline without vendor
documentation, licensing confirmation, packet validation, and device tests.

## Pass criteria for the next run

Gate 1 may be marked PASS only when all three exported pre-inference frames:

- contain the same valid raw USG content (not a black secure surface),
- contain no cyan box, mask, label, centroid/depth graphic, or HUD,
- remain clean while the corresponding graphics are visibly rendered on the
  physical SM-T295.

Until then, model export, detector A/B evaluation, state-machine tuning, coast
window tuning, and display smoothing remain intentionally unexecuted.
