"""
SAM2 mask-only predictions — uses mask area changes to predict stages and outcome.

Logic:
  - Mask area increasing → object being grasped/lifted
  - Mask area stable → object held/moving
  - Mask area decreasing sharply → object dropped or placed
  - Mask disappears → object left frame or placed in container
"""
import json
import pickle
import numpy as np
from pathlib import Path

SAMPLES  = json.load(open("../robot-task-monitoring/data/samples.json"))
OUT_FILE = Path("predictions_masks.jsonl")


def load_masks(pkl_path: str) -> np.ndarray:
    """Load SAM2 masks from pkl — same logic as mask_loader.py."""
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, np.ndarray):
        return obj.astype(bool)
    elif isinstance(obj, dict):
        arrays = []
        for key, val in obj.items():
            if isinstance(val, np.ndarray):
                arrays.append(val.astype(bool))
            elif isinstance(val, list):
                arrays.append(np.array(val, dtype=bool))
        if arrays:
            return np.logical_or.reduce(arrays)
    elif isinstance(obj, list):
        return np.array(obj, dtype=bool)

    raise ValueError(f"Unknown pkl format: {type(obj)}")


def predict_from_masks(vid: str, masks: np.ndarray, fps: float = 30.0) -> dict:
    T = len(masks)
    clip_s = T / fps

    # Compute per-frame mask area (fraction of image)
    total_pixels = masks[0].size if masks.ndim >= 2 else 1
    areas = np.array([m.sum() / total_pixels for m in masks])

    # Smooth the area curve
    kernel_size = min(5, T)
    kernel = np.ones(kernel_size) / kernel_size
    areas_smooth = np.convolve(areas, kernel, mode="same")

    # Compute area derivatives
    area_diff = np.diff(areas_smooth)

    # Find key events
    GROW_THRESHOLD  = 0.005   # area growing = object being engaged
    SHRINK_THRESHOLD = -0.005  # area shrinking = object leaving

    # Find first significant growth (grasp)
    grow_frames = np.where(area_diff > GROW_THRESHOLD)[0]
    grasp_time = grow_frames[0] / fps if len(grow_frames) > 0 else None

    # Find significant shrink after growth (drop or place)
    shrink_frames = np.where(area_diff < SHRINK_THRESHOLD)[0]
    if grasp_time is not None and len(shrink_frames) > 0:
        post_grasp_shrinks = shrink_frames[shrink_frames > (grasp_time * fps + 5)]
        release_time = post_grasp_shrinks[0] / fps if len(post_grasp_shrinks) > 0 else None
    else:
        release_time = None

    # Final mask area vs initial
    initial_area = areas_smooth[:5].mean() if T > 5 else areas_smooth[0]
    final_area   = areas_smooth[-5:].mean() if T > 5 else areas_smooth[-1]
    area_change  = final_area - initial_area

    # Peak area
    peak_area_idx = np.argmax(areas_smooth)
    peak_area_time = peak_area_idx / fps

    # --- Stage detection ---
    stages = []
    stages.append({"name": "approach", "start_s": 0.0, "end_s": round(grasp_time or clip_s * 0.3, 2)})

    if grasp_time is not None:
        stages.append({"name": "grasp", "start_s": round(grasp_time, 2),
                        "end_s": round(grasp_time + 0.4, 2)})
        lift_start = grasp_time + 0.4
        stages.append({"name": "lift", "start_s": round(lift_start, 2),
                        "end_s": round(min(lift_start + 0.5, clip_s), 2)})

        if release_time is not None:
            stages.append({"name": "move", "start_s": round(lift_start + 0.5, 2),
                            "end_s": round(release_time, 2)})
            stages.append({"name": "place", "start_s": round(release_time, 2),
                            "end_s": round(min(release_time + 0.5, clip_s), 2)})
        else:
            stages.append({"name": "move", "start_s": round(lift_start + 0.5, 2),
                            "end_s": round(clip_s, 2)})
    else:
        stages.append({"name": "idle", "start_s": round(clip_s * 0.3, 2), "end_s": round(clip_s, 2)})

    # --- Outcome decision ---
    # If mask area returns to near-initial and release was smooth → success
    area_returned = abs(area_change) < 0.02

    if grasp_time is not None and release_time is not None and area_returned:
        outcome = "success"
        failure_cause = "none"
        confidence = 0.60
        explanation = (f"Mask area grew at t={grasp_time:.2f}s (grasp), shrank at "
                       f"t={release_time:.2f}s (release), returned to baseline. "
                       f"Consistent with pick-and-place success.")
    elif grasp_time is not None and release_time is not None and not area_returned:
        outcome = "failure"
        failure_cause = "incorrect_placement"
        confidence = 0.50
        explanation = (f"Object grasped at t={grasp_time:.2f}s and released at "
                       f"t={release_time:.2f}s, but mask area didn't return to baseline "
                       f"(delta={area_change:.3f}). Object may be misplaced.")
    elif grasp_time is not None and release_time is None:
        outcome = "failure"
        failure_cause = "unstable_movement"
        confidence = 0.45
        explanation = (f"Grasp detected at t={grasp_time:.2f}s but no release event. "
                       f"Object may still be held or was lost.")
    else:
        outcome = "failure"
        failure_cause = "missed_grasp"
        confidence = 0.40
        explanation = f"No significant mask area change detected. Mean area: {areas.mean():.4f}."

    return {
        "video_id": vid,
        "stages": stages,
        "outcome": outcome,
        "failure_cause": failure_cause,
        "confidence": confidence,
        "evidence": {
            "raft_vertical_drop_at_s": 0.0,
            "depth_gap_to_target_mm": 0.0,
            "explanation": f"[Mask-only] {explanation}"
        }
    }


# --- Run ---
predictions = []
for sample in SAMPLES:
    vid = sample["video_id"]
    pkl_path = sample.get("sam2_pkl_path")
    if not pkl_path or not Path(pkl_path).exists():
        print(f"  ✗ {vid}: no mask file, skipping")
        continue
    try:
        masks = load_masks(pkl_path)
        pred = predict_from_masks(vid, masks)
        predictions.append(pred)
        print(f"  ✓ {vid}: outcome={pred['outcome']}, cause={pred['failure_cause']}, conf={pred['confidence']:.2f}")
    except Exception as e:
        print(f"  ✗ {vid}: error — {e}")

OUT_FILE.write_text("\n".join(json.dumps(p) for p in predictions) + "\n")
print(f"\n→ Saved {len(predictions)} predictions to {OUT_FILE}")
