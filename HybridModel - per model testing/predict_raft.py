"""
RAFT-only predictions — uses optical flow signals to predict stages and outcome.

Logic:
  - lift_onset_s exists → grasp happened
  - place_onset_s exists → placement happened
  - Both exist → success
  - peak_vertical_drop > threshold → dropped_object
  - No lift_onset → missed_grasp
"""
import json
from pathlib import Path

MOTION_DIR = Path("../robot-task-monitoring/outputs/motion")
SAMPLES    = json.load(open("../robot-task-monitoring/data/samples.json"))
OUT_FILE   = Path("predictions_raft.jsonl")


def predict_from_raft(vid: str, motion: dict) -> dict:
    fps       = motion.get("fps", 30.0)
    n_frames  = motion.get("n_frames", 121)
    clip_s    = n_frames / fps

    lift      = motion.get("lift_onset_s")
    place     = motion.get("place_onset_s")
    peak_drop = motion.get("peak_vertical_drop", 0)
    peak_time = motion.get("peak_time_s", 0)
    activity  = motion.get("activity_ratio", 0)
    mag_mean  = motion.get("magnitude", {}).get("mean", 0)

    # --- Stage detection from motion events ---
    stages = []
    
    # Approach: start of clip until first significant motion
    approach_end = lift if lift else clip_s * 0.3
    stages.append({"name": "approach", "start_s": 0.0, "end_s": round(approach_end, 2)})

    if lift is not None:
        # Align phase: brief period before lift
        align_start = max(0, lift - 0.3)
        stages.append({"name": "align", "start_s": round(align_start, 2), "end_s": round(lift, 2)})

        # Grasp phase
        grasp_end = lift + 0.3
        stages.append({"name": "grasp", "start_s": round(lift, 2), "end_s": round(grasp_end, 2)})

        # Lift phase
        lift_end = grasp_end + 0.5
        stages.append({"name": "lift", "start_s": round(grasp_end, 2), "end_s": round(min(lift_end, clip_s), 2)})

        if place is not None:
            # Move phase: from lift end to place onset
            stages.append({"name": "move", "start_s": round(lift_end, 2), "end_s": round(place, 2)})
            # Place phase
            place_end = min(place + 0.5, clip_s)
            stages.append({"name": "place", "start_s": round(place, 2), "end_s": round(place_end, 2)})
        else:
            # Move but no place — motion continues until clip ends
            stages.append({"name": "move", "start_s": round(lift_end, 2), "end_s": round(clip_s, 2)})
    else:
        # No lift detected — idle after approach
        stages.append({"name": "idle", "start_s": round(approach_end, 2), "end_s": round(clip_s, 2)})

    # --- Outcome decision ---
    DROP_THRESHOLD = 20.0  # px/frame — tunable

    if lift is not None and place is not None:
        outcome = "success"
        failure_cause = "none"
        confidence = 0.75
        explanation = (f"Lift detected at t={lift:.2f}s, placement at t={place:.2f}s. "
                       f"Full grasp-to-place cycle completed.")
    elif lift is not None and peak_drop > DROP_THRESHOLD:
        outcome = "failure"
        failure_cause = "dropped_object"
        confidence = 0.70
        explanation = (f"Lift at t={lift:.2f}s but peak vertical drop of {peak_drop:.1f} px/frame "
                       f"at t={peak_time:.2f}s suggests object was dropped.")
    elif lift is not None and place is None:
        outcome = "failure"
        failure_cause = "unstable_movement"
        confidence = 0.55
        explanation = (f"Lift at t={lift:.2f}s but no placement detected. "
                       f"Object may have been lost during transport.")
    elif activity < 0.15:
        outcome = "failure"
        failure_cause = "timeout"
        confidence = 0.60
        explanation = f"Very low activity ratio ({activity:.2f}). Robot appears idle."
    else:
        outcome = "failure"
        failure_cause = "missed_grasp"
        confidence = 0.50
        explanation = (f"No lift onset detected. Mean motion magnitude: {mag_mean:.2f} px/frame. "
                       f"Grasp likely failed.")

    return {
        "video_id": vid,
        "stages": stages,
        "outcome": outcome,
        "failure_cause": failure_cause,
        "confidence": confidence,
        "evidence": {
            "raft_vertical_drop_at_s": round(peak_time, 4),
            "depth_gap_to_target_mm": 0.0,
            "explanation": f"[RAFT-only] {explanation}"
        }
    }


# --- Run ---
predictions = []
for sample in SAMPLES:
    vid = sample["video_id"]
    motion_file = MOTION_DIR / f"{vid}.json"
    if not motion_file.exists():
        print(f"  ✗ {vid}: no motion file, skipping")
        continue
    motion = json.load(open(motion_file))
    pred = predict_from_raft(vid, motion)
    predictions.append(pred)
    print(f"  ✓ {vid}: outcome={pred['outcome']}, cause={pred['failure_cause']}, conf={pred['confidence']:.2f}")

OUT_FILE.write_text("\n".join(json.dumps(p) for p in predictions) + "\n")
print(f"\n→ Saved {len(predictions)} predictions to {OUT_FILE}")
