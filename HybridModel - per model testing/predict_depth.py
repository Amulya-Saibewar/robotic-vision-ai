"""
Depth-only predictions — uses depth map features to predict stages and outcome.

Logic:
  - Small min_gripper_to_object_distance → gripper reached object → grasp attempted
  - Small depth_gap_to_target_mm → object reached target → success
  - Large depth_gap → object didn't reach target → failure
  - depth_gradient_at_grasp direction → approach vs retreat
"""
import json
from pathlib import Path

DEPTH_DIR = Path("../robot-task-monitoring/outputs/depth")
SAMPLES   = json.load(open("../robot-task-monitoring/data/samples.json"))
OUT_FILE  = Path("predictions_depth.jsonl")


def predict_from_depth(vid: str, depth: dict) -> dict:
    unit            = depth.get("unit", "m")
    min_dist        = depth.get("min_gripper_to_object_distance", 999)
    min_dist_time   = depth.get("min_distance_time_s", 0)
    gradient        = depth.get("depth_gradient_at_grasp", 0)
    depth_gap       = depth.get("depth_gap_to_target_mm", 999)
    obj_depth_mean  = depth.get("object_depth_mean", 0)
    n_frames        = depth.get("n_frames", 121)
    fps             = depth.get("fps", 30.0)
    clip_s          = n_frames / fps if n_frames and fps else 4.0

    # Thresholds
    GRASP_DIST_THRESHOLD = 50.0    # mm — gripper close enough to attempt grasp
    TARGET_GAP_THRESHOLD = 20.0    # mm — close enough to target = success
    
    # Convert if needed
    if unit == "m":
        min_dist_mm = min_dist * 1000
    else:
        min_dist_mm = min_dist

    # --- Stage detection ---
    stages = []
    grasp_attempted = min_dist_mm < GRASP_DIST_THRESHOLD

    # Approach phase: start to min-distance time
    approach_end = min_dist_time if min_dist_time > 0 else clip_s * 0.3
    stages.append({"name": "approach", "start_s": 0.0, "end_s": round(approach_end, 2)})

    if grasp_attempted:
        # Align: just before closest approach
        align_start = max(0, min_dist_time - 0.3)
        stages.append({"name": "align", "start_s": round(align_start, 2), "end_s": round(min_dist_time, 2)})

        # Grasp: at closest approach
        grasp_end = min_dist_time + 0.3
        stages.append({"name": "grasp", "start_s": round(min_dist_time, 2), "end_s": round(grasp_end, 2)})

        if gradient < -0.01:
            # Negative gradient = object moving away from camera = lifting
            lift_end = grasp_end + 0.5
            stages.append({"name": "lift", "start_s": round(grasp_end, 2), "end_s": round(min(lift_end, clip_s), 2)})
            stages.append({"name": "move", "start_s": round(lift_end, 2), "end_s": round(clip_s * 0.85, 2)})

            if depth_gap < TARGET_GAP_THRESHOLD:
                stages.append({"name": "place", "start_s": round(clip_s * 0.85, 2), "end_s": round(clip_s, 2)})
        else:
            stages.append({"name": "idle", "start_s": round(grasp_end, 2), "end_s": round(clip_s, 2)})
    else:
        stages.append({"name": "idle", "start_s": round(approach_end, 2), "end_s": round(clip_s, 2)})

    # --- Outcome decision ---
    if grasp_attempted and depth_gap < TARGET_GAP_THRESHOLD:
        outcome = "success"
        failure_cause = "none"
        confidence = 0.70
        explanation = (f"Gripper reached object (min distance: {min_dist_mm:.1f}mm at t={min_dist_time:.2f}s). "
                       f"Final depth gap to target: {depth_gap:.1f}mm — within threshold.")
    elif grasp_attempted and depth_gap >= TARGET_GAP_THRESHOLD:
        outcome = "failure"
        failure_cause = "incorrect_placement"
        confidence = 0.60
        explanation = (f"Gripper reached object ({min_dist_mm:.1f}mm) but final depth gap "
                       f"is {depth_gap:.1f}mm — object didn't reach target.")
    elif not grasp_attempted:
        outcome = "failure"
        failure_cause = "missed_grasp"
        confidence = 0.55
        explanation = (f"Minimum gripper-to-object distance was {min_dist_mm:.1f}mm — "
                       f"too far for a successful grasp attempt.")
    else:
        outcome = "failure"
        failure_cause = "unstable_movement"
        confidence = 0.45
        explanation = f"Ambiguous depth signals. Gap: {depth_gap:.1f}mm."

    return {
        "video_id": vid,
        "stages": stages,
        "outcome": outcome,
        "failure_cause": failure_cause,
        "confidence": confidence,
        "evidence": {
            "raft_vertical_drop_at_s": 0.0,
            "depth_gap_to_target_mm": round(depth_gap, 2),
            "explanation": f"[Depth-only] {explanation}"
        }
    }


# --- Run ---
predictions = []
for sample in SAMPLES:
    vid = sample["video_id"]
    depth_file = DEPTH_DIR / f"{vid}.json"
    if not depth_file.exists():
        print(f"  ✗ {vid}: no depth file, skipping")
        continue
    depth = json.load(open(depth_file))
    pred = predict_from_depth(vid, depth)
    predictions.append(pred)
    print(f"  ✓ {vid}: outcome={pred['outcome']}, cause={pred['failure_cause']}, conf={pred['confidence']:.2f}")

OUT_FILE.write_text("\n".join(json.dumps(p) for p in predictions) + "\n")
print(f"\n→ Saved {len(predictions)} predictions to {OUT_FILE}")
