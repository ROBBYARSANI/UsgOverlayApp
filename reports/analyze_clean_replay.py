"""Generate deterministic clean-replay metrics and diagnostic videos."""

import csv
import json
import math
import statistics
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
VIDEO = Path(r"D:\D4 TEKNIK KOMPUTER\TA\Data_Citra_USG\Subjek_1\WirelessUSG2026-07-29-16-12-25.mp4")
OLD_CSV = REPORTS / "raw_yolo_video_detections.csv"
HARD_CSV = REPORTS / "raw_yolo_video_detections_hardneg.csv"
GT_CSV = REPORTS / "tracking_trajectories_comparison.csv"
MATCH_RADIUS_PX = 30.0


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def truthy(value):
    return str(value).lower() == "true"


def detection(row):
    if not truthy(row["detection_present"]):
        return None
    left, top = float(row["left"]), float(row["top"])
    right, bottom = float(row["right"]), float(row["bottom"])
    return {
        "left": left, "top": top, "right": right, "bottom": bottom,
        "cx": (left + right) * 0.5, "cy": (top + bottom) * 0.5,
        "confidence": float(row["confidence"]),
    }


def streak_lengths(values, wanted):
    lengths = []
    current = 0
    for value in values:
        if value == wanted:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def classify(row, gt):
    det = detection(row)
    valid = bool(gt["gt_cx"] and gt["gt_cy"])
    if det is None:
        return ("MISS" if valid else "TN"), None
    if not valid:
        return "FP", None
    error = math.hypot(det["cx"] - float(gt["gt_cx"]),
                       det["cy"] - float(gt["gt_cy"]))
    return ("TP" if error <= MATCH_RADIUS_PX else "WRONG_TARGET"), error


def summarize(rows, gt_rows):
    present = [truthy(row["detection_present"]) for row in rows]
    classes = [classify(row, gt)[0] for row, gt in zip(rows, gt_rows)]
    errors = [classify(row, gt)[1] for row, gt in zip(rows, gt_rows)]
    good_errors = [value for value in errors if value is not None and value <= MATCH_RADIUS_PX]
    fp = classes.count("FP") + classes.count("WRONG_TARGET")
    fn = classes.count("MISS") + classes.count("WRONG_TARGET")
    tp = classes.count("TP")
    tn = classes.count("TN")
    intervals = [float(row.get("fresh_inference_interval_ms") or 0.0) for row in rows]
    intervals = [value for value in intervals if value > 0.0]
    completion = [float(row.get("completion_elapsed_ms") or 0.0) for row in rows]
    detected_completion = [time for time, yes in zip(completion, present) if yes and time > 0.0]
    detected_intervals = [b - a for a, b in zip(detected_completion, detected_completion[1:])]
    miss_streaks = streak_lengths(present, False)
    det_streaks = streak_lengths(present, True)

    # A blink is a detector absence while the independent reference says target remains valid.
    false_absent = [valid and not yes for valid, yes in zip(
        [bool(g["gt_cx"] and g["gt_cy"]) for g in gt_rows], present)]
    blink_events = len(streak_lengths(false_absent, True))
    return {
        "completed_inferences": len(rows),
        "detections": sum(present),
        "presence_rate": sum(present) / len(rows),
        "detection_streaks": det_streaks,
        "miss_streaks": miss_streaks,
        "longest_detection_streak": max(det_streaks, default=0),
        "longest_miss_streak": max(miss_streaks, default=0),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "false_blink_events": blink_events,
        "mean_centroid_error_px": statistics.mean(good_errors) if good_errors else None,
        "mean_confidence": statistics.mean(float(row["confidence"]) for row in rows
                                             if truthy(row["detection_present"])),
        "mean_mask_foreground_ratio": statistics.mean(
            float(row["mask_foreground_ratio"]) for row in rows
            if truthy(row["detection_present"])),
        "mean_latency_ms": statistics.mean(float(row["total_latency_ms"]) for row in rows),
        "median_latency_ms": statistics.median(float(row["total_latency_ms"]) for row in rows),
        "mean_fresh_inference_interval_ms": statistics.mean(intervals) if intervals else None,
        "median_fresh_inference_interval_ms": statistics.median(intervals) if intervals else None,
        "mean_fresh_detection_interval_ms": statistics.mean(detected_intervals)
        if detected_intervals else None,
    }


def write_ab_csv(old_rows, hard_rows, gt_rows):
    output = REPORTS / "raw_model_ab_comparison.csv"
    fields = [
        "frame_index", "video_timestamp_ms", "frame_hash_match", "gt_valid", "gt_cx", "gt_cy",
        "old_status", "old_confidence", "old_left", "old_top", "old_right", "old_bottom",
        "old_centroid_error_px", "old_mask_present", "old_mask_foreground_ratio", "old_latency_ms",
        "hardneg_status", "hardneg_confidence", "hardneg_left", "hardneg_top", "hardneg_right",
        "hardneg_bottom", "hardneg_centroid_error_px", "hardneg_mask_present",
        "hardneg_mask_foreground_ratio", "hardneg_latency_ms",
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for old, hard, gt in zip(old_rows, hard_rows, gt_rows):
            old_status, old_error = classify(old, gt)
            hard_status, hard_error = classify(hard, gt)
            writer.writerow({
                "frame_index": old["frame_index"],
                "video_timestamp_ms": old["video_timestamp_ms"],
                "frame_hash_match": old["frame_sha256"] == hard["frame_sha256"],
                "gt_valid": bool(gt["gt_cx"] and gt["gt_cy"]),
                "gt_cx": gt["gt_cx"], "gt_cy": gt["gt_cy"],
                "old_status": old_status, "old_confidence": old["confidence"],
                "old_left": old["left"], "old_top": old["top"],
                "old_right": old["right"], "old_bottom": old["bottom"],
                "old_centroid_error_px": "" if old_error is None else f"{old_error:.4f}",
                "old_mask_present": old["mask_present"],
                "old_mask_foreground_ratio": old["mask_foreground_ratio"],
                "old_latency_ms": old["total_latency_ms"],
                "hardneg_status": hard_status, "hardneg_confidence": hard["confidence"],
                "hardneg_left": hard["left"], "hardneg_top": hard["top"],
                "hardneg_right": hard["right"], "hardneg_bottom": hard["bottom"],
                "hardneg_centroid_error_px": "" if hard_error is None else f"{hard_error:.4f}",
                "hardneg_mask_present": hard["mask_present"],
                "hardneg_mask_foreground_ratio": hard["mask_foreground_ratio"],
                "hardneg_latency_ms": hard["total_latency_ms"],
            })
    return output


def simulate_coast(rows, gt_rows, factor, median_interval):
    coast_ms = factor * median_interval
    state = "SEARCH"
    last_fresh_time = None
    visible = False
    first_miss_seen = False
    timeline = []
    completion = [float(row["completion_elapsed_ms"]) for row in rows]
    for row, now in zip(rows, completion):
        det = truthy(row["detection_present"])
        if det:
            state, visible = "LOCKED", True
            last_fresh_time = now
            first_miss_seen = False
        elif state == "LOCKED":
            state, visible = "COAST", True
            first_miss_seen = True
        elif state == "COAST":
            age = now - last_fresh_time
            if first_miss_seen and age > coast_ms:
                state, visible = "LOST", False
            else:
                visible = True
        elif state == "SEARCH":
            visible = False
        else:
            state, visible = "LOST", False
        timeline.append((now, state, visible))

    gt_valid = [bool(g["gt_cx"] and g["gt_cy"]) for g in gt_rows]
    false_hidden = [valid and not item[2] for valid, item in zip(gt_valid, timeline)]
    false_visible = [(not valid) and item[2] for valid, item in zip(gt_valid, timeline)]

    def run_durations(flags):
        result, start = [], None
        for i, flag in enumerate(flags):
            if flag and start is None:
                start = i
            if start is not None and (not flag or i == len(flags) - 1):
                end = i if flag and i == len(flags) - 1 else i - 1
                # State chosen at completion[start] remains active until the completion
                # immediately after the run. Measure that display-active interval.
                after = completion[end + 1] if end + 1 < len(completion) else completion[end]
                result.append(max(0.0, after - completion[start]))
                start = None
        return result

    hidden = run_durations(false_hidden)
    ghosts = run_durations(false_visible)
    reacquisition = []
    for i in range(1, len(gt_valid)):
        if gt_valid[i] and not gt_valid[i - 1]:
            acquired = next((completion[j] - completion[i] for j in range(i, len(rows))
                             if timeline[j][2]), None)
            if acquired is not None:
                reacquisition.append(max(0.0, acquired))
    return {
        "factor": factor,
        "coast_ms": coast_ms,
        "blink_count": len(hidden),
        "max_hidden_ms": max(hidden, default=0.0),
        "ghost_events": len(ghosts),
        "max_ghost_ms": max(ghosts, default=0.0),
        "total_ghost_ms": sum(ghosts),
        "mean_reacquisition_ms": statistics.mean(reacquisition) if reacquisition else 0.0,
    }


def interpolation_metrics(rows, duration_ms):
    centers = []
    for row in rows:
        det = detection(row)
        if det:
            centers.append((det["cx"], det["cy"]))
    jumps = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(centers, centers[1:])]
    display_steps = []
    max_lag = 0.0
    for jump in jumps:
        if duration_ms <= 0 or jump > 110.0:
            display_steps.append(jump)
            continue
        count = max(1, math.ceil(duration_ms / (1000.0 / 60.0)))
        previous = 0.0
        for tick in range(1, count + 1):
            p = min(1.0, tick / count)
            eased = p * p * (3.0 - 2.0 * p)
            display_steps.append(jump * (eased - previous))
            max_lag = max(max_lag, jump * (1.0 - eased))
            previous = eased
    return {
        "duration_ms": duration_ms,
        "max_display_step_px": max(display_steps, default=0.0),
        "mean_display_step_px": statistics.mean(display_steps) if display_steps else 0.0,
        "display_steps_over_10px": sum(step > 10.0 for step in display_steps),
        "max_visual_lag_px": max_lag,
        "detector_coordinates_changed": False,
    }


def render_raw(rows, output):
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (width, height))
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        row = rows[index]
        det = detection(row)
        if det:
            cv2.rectangle(frame, (round(det["left"]), round(det["top"])),
                          (round(det["right"]), round(det["bottom"])), (255, 240, 0), 3)
            cv2.putText(frame, f"RAW {det['confidence']:.3f}",
                        (round(det["left"]), max(25, round(det["top"]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 240, 0), 2, cv2.LINE_AA)
        writer.write(frame)
        index += 1
    writer.release()
    cap.release()


def main():
    old_rows, hard_rows, gt_rows = read_csv(OLD_CSV), read_csv(HARD_CSV), read_csv(GT_CSV)
    if not (len(old_rows) == len(hard_rows) == len(gt_rows) == 199):
        raise RuntimeError("Expected 199 aligned rows")
    hash_mismatches = sum(a["frame_sha256"] != b["frame_sha256"]
                          for a, b in zip(old_rows, hard_rows))
    old_summary = summarize(old_rows, gt_rows)
    hard_summary = summarize(hard_rows, gt_rows)
    interval = old_summary["median_fresh_inference_interval_ms"]
    if interval is None:
        raise RuntimeError("Timed baseline replay is required")

    coast = [simulate_coast(old_rows, gt_rows, factor, interval)
             for factor in (1.0, 1.5, 2.0)]
    interpolation = [interpolation_metrics(old_rows, duration)
                     for duration in (0, 50, 75, 100)]
    metrics = {
        "source_video": str(VIDEO),
        "device": "SM-T295 / Android 11",
        "frame_hash_mismatches": hash_mismatches,
        "ground_truth_note": "Existing project morphology-assisted reference; 30 px match gate.",
        "old_model": old_summary,
        "hardnegative_model": hard_summary,
        "selected_model": "YoloV8_320.tflite",
        "coast_trials": coast,
        "selected_coast_factor": 1.0,
        "interpolation_trials": interpolation,
        "selected_interpolation_ms": 75,
    }
    (REPORTS / "clean_replay_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    with (REPORTS / "antiblink_coast_comparison.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(coast[0].keys()))
        writer.writeheader()
        writer.writerows(coast)
    with (REPORTS / "display_interpolation_comparison.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(interpolation[0].keys()))
        writer.writeheader()
        writer.writerows(interpolation)
    write_ab_csv(old_rows, hard_rows, gt_rows)
    render_raw(old_rows, REPORTS / "raw_yolo_replay_output.mp4")
    render_raw(old_rows, REPORTS / "raw_yolo_replay_output_old.mp4")
    render_raw(hard_rows, REPORTS / "raw_yolo_replay_output_hardneg.mp4")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
