"""
pick_place_cube × Cosmos-Reason2-2B  — FIXED Pipeline
=======================================================
Dataset : theconstruct-ai/pick_place_cube
Model   : nvidia/Cosmos-Reason2-2B

GT derivation fix:
  SUCCESS = robot closed gripper (grasp) AND lifted (z rose) AND
            then opened gripper again (place/release) — full pick-and-place cycle
  FAILURE = any step missing: no grasp, no lift, or no release after lift

Cosmos prompt fix:
  Raw sensor numbers are shown neutrally — no pre-labelled icons that bias the model.
  Cosmos makes its own visual + sensor judgement.

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxx
    pip3 install pandas pyarrow
    python3 pick_place_pipeline.py
"""

import os, json, re, traceback
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
HF_TOKEN        = os.getenv("HF_TOKEN", "")
DATASET_REPO    = "theconstruct-ai/pick_place_cube"
MODEL_ID        = "nvidia/Cosmos-Reason2-2B"
OUTPUT_DIR      = Path("outputs")
PRED_DIR        = OUTPUT_DIR / "predictions"
NUM_EPISODES    = 21
NUM_FRAMES      = 16          # 8 overhead + 8 wrist
MAX_NEW_TOKENS  = 1024
FPS             = 15.0
TASK_DESC       = "pick the green cube and place it"

# ── GT thresholds (carefully tuned for pick-and-place) ────────────────────────
# Success requires ALL of:
Z_LIFT_MIN          = 0.04    # z must rise at least 4cm (cube is being lifted)
GRIPPER_CLOSE_THRESH = 0.5    # action[3] < 0.5 → gripper closed
MIN_CLOSED_FRAMES   = 5       # gripper must stay closed for at least N frames
MIN_GRASP_EVENTS    = 1       # at least one close event
MIN_RELEASE_EVENTS  = 1       # at least one open-after-close event (placement)
# ──────────────────────────────────────────────────────────────────────────────

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login, hf_hub_download


# ══════════════════════════════════════════════════════════════════════════════
# PROMPTS  — neutral, no pre-baked labels that bias Cosmos
# ══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an expert physical-AI analyst specialising in robotic manipulation.

You are given frames from TWO camera views of a robot arm performing a pick-and-place task:
  • CAM_OVERHEAD (images 1–8)  : wide scene view — robot arm, table, green cube
  • CAM_WRIST   (images 9–16) : close-up from robot wrist — gripper and cube interaction

You are also given a SENSOR DATA REPORT with robot state telemetry.

TASK: "Pick the green cube from the table and place it at a target location."
SUCCESS means: the robot grasped the cube, lifted it off the table, moved it, and released it at a new location.
FAILURE means: the robot did NOT complete the full pick-and-place — e.g. missed the cube, dropped it mid-air, never grasped it, or never released it at a target.

Your analysis must:
1. Carefully examine both camera views to visually determine what actually happened
2. Cross-reference the sensor data to confirm or question the visual evidence
3. Identify the sequential stages actually observed
4. Determine final_result: "success" or "failure"
5. If failure: explain PRECISELY what went wrong, at which stage, and why
6. Provide a full description of everything happening in the video
7. Give a confidence score 0.0–1.0 for your classification

IMPORTANT RULES:
- A gripper opening at the END (after lift + move) is PLACEMENT, not failure
- A gripper opening DURING lift (cube drops mid-air) IS failure  
- The gripper being open at the START (before grasp) is normal idle/approach
- Judge success only if the FULL sequence (grasp → lift → move → place) is visible
- Do NOT blindly trust the sensor data — use your visual judgement as primary signal

Reply with ONLY a valid JSON object — no markdown fences, no extra text:
{
  "video_id": "<episode filename>",
  "task_stages": ["<stage1>", "<stage2>", ...],
  "final_result": "success" or "failure",
  "failure_reason": "<ONLY include this field if final_result is failure — precise explanation of what failed, at which stage, what the robot did wrong>",
  "reasoning": "<comprehensive description of everything observed — object positions, gripper actions, lift height, placement, outcome>",
  "confidence": <float 0.0 to 1.0>
}"""
 
 
def build_user_prompt(ep_idx: int, f: dict) -> str:
    """
    Build a neutral, richly-informative sensor data report.
    No pre-labelled success/failure icons — Cosmos judges for itself.
    """
    return f"""EPISODE: episode_{ep_idx:06d}.mp4
TASK: {TASK_DESC}
FPS: {FPS}  |  TOTAL FRAMES: {f['total_frames']}  |  DURATION: {f['duration_sec']:.1f}s
 
━━━ SENSOR DATA REPORT ━━━
 
[END-EFFECTOR TRAJECTORY]
  Start position (x, y, z) : {f['pos_start']}  metres
  End position   (x, y, z) : {f['pos_end']}  metres
  Peak Z height reached     : {f['peak_z']:.4f} m
  Z change (end − start)    : {f['z_delta']:+.4f} m
  Max Z rise from baseline  : {f['z_rise']:.4f} m
  Total XY displacement     : {f['total_xy_dist']:.4f} m
  Total 3D path length      : {f['total_path_len']:.4f} m
 
[VELOCITY PROFILE]
  Mean step velocity : {f['mean_vel']:.5f} m/frame
  Max  step velocity : {f['max_vel']:.5f} m/frame
  Velocity std dev   : {f['vel_std']:.5f}
  High-speed frames (>mean+2σ): {f['abrupt_count']}
  Motion character   : {f['motion_quality']}
 
[GRIPPER STATE TIMELINE]
  Gripper encoding: action[3] value — 1.0 = fully open, 0.0 = fully closed
  Frames with gripper OPEN   (>0.5): {f['gripper_open_frames']}  ({f['gripper_open_pct']:.1f}%)
  Frames with gripper CLOSED (<0.5): {f['gripper_closed_frames']}  ({f['gripper_closed_pct']:.1f}%)
  Open→Close transitions (grasps)  : {f['grasp_events']}
  Close→Open transitions (releases): {f['release_events']}
  First CLOSE event at frame        : {f['first_close_frame']}  (t = {f['first_close_time']:.2f}s)
  First RELEASE after grasp frame   : {f['first_release_after_grasp_frame']}  (t = {f['first_release_after_grasp_time']:.2f}s)
  Longest closed-gripper run        : {f['max_closed_run']} consecutive frames
  Final gripper state               : {'CLOSED (value < 0.5)' if f['final_gripper_closed'] else 'OPEN (value >= 0.5)'}
 
[Z-HEIGHT PROFILE over episode]
  Z at  0% (start) : {f['z_pct'][0]:.4f} m
  Z at 25%         : {f['z_pct'][1]:.4f} m
  Z at 50% (mid)   : {f['z_pct'][2]:.4f} m
  Z at 75%         : {f['z_pct'][3]:.4f} m
  Z at 100% (end)  : {f['z_pct'][4]:.4f} m
  Z profile shape  : {f['z_shape']}
 
[PICK-AND-PLACE CYCLE ANALYSIS]
  Phase 1 — Approach & Descend : Z trend before first grasp = {f['z_before_grasp_trend']}
  Phase 2 — Grasp              : gripper closed for {f['max_closed_run']} frames, first at t={f['first_close_time']:.2f}s
  Phase 3 — Lift               : max Z rise after grasp = {f['z_rise_after_grasp']:.4f} m
  Phase 4 — Move & Place       : release events after grasp = {f['release_events']}
  Phase 5 — Retreat            : Z at end = {f['z_pct'][4]:.4f} m
 
━━━ VISUAL FRAMES FOLLOW ━━━
Images 1–8  = CAM_OVERHEAD (wide scene view)
Images 9–16 = CAM_WRIST   (close-up gripper view)
 
Use visual evidence as your PRIMARY signal. Use sensor data to confirm.
Return the JSON now."""


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION — corrected pick-and-place logic
# ══════════════════════════════════════════════════════════════════════════════
def extract_features(ep_idx: int) -> dict:
    parquet_path = f"data/chunk-000/episode_{ep_idx:06d}.parquet"
    local = hf_hub_download(DATASET_REPO, parquet_path, repo_type="dataset", token=HF_TOKEN)
    df = pd.read_parquet(local)

    states  = np.array(df["observation.state"].tolist())  # (N,3) x,y,z
    actions = np.array(df["action"].tolist())              # (N,4) dx,dy,dz,gripper
    times   = df["timestamp"].values
    N       = len(df)

    x, y, z  = states[:, 0], states[:, 1], states[:, 2]
    gripper  = actions[:, 3]   # 1.0=open, 0.0=closed

    # ── Velocity ──────────────────────────────────────────────────────────────
    diffs     = np.diff(states, axis=0)
    step_dist = np.linalg.norm(diffs, axis=1)
    xy_diffs  = np.diff(np.stack([x, y], axis=1), axis=0)
    xy_dist   = np.linalg.norm(xy_diffs, axis=1)

    mean_vel = float(np.mean(step_dist))
    max_vel  = float(np.max(step_dist))
    vel_std  = float(np.std(step_dist))
    abrupt   = int(np.sum(step_dist > mean_vel + 2 * vel_std))

    if vel_std < 0.001:
        motion_quality = "STATIC — robot barely moved"
    elif abrupt > 15:
        motion_quality = "ERRATIC — frequent abrupt velocity changes"
    elif max_vel > 0.05:
        motion_quality = "DYNAMIC — fast with some speed spikes"
    else:
        motion_quality = "SMOOTH — controlled and steady"

    # ── Gripper ───────────────────────────────────────────────────────────────
    closed_mask = gripper < GRIPPER_CLOSE_THRESH   # True = closed

    open_frames   = int(np.sum(~closed_mask))
    closed_frames = int(np.sum(closed_mask))
    open_pct      = 100.0 * open_frames / N
    closed_pct    = 100.0 * closed_frames / N

    transitions    = np.diff(closed_mask.astype(int))
    grasp_events   = int(np.sum(transitions == 1))   # open→close
    release_events = int(np.sum(transitions == -1))  # close→open

    # First close
    first_close_idx  = int(np.argmax(closed_mask)) if closed_frames > 0 else N - 1
    first_close_time = float(times[first_close_idx])

    # First release AFTER first close
    release_idxs = np.where(transitions == -1)[0] + 1  # indices where close→open
    post_grasp_releases = release_idxs[release_idxs > first_close_idx]
    if len(post_grasp_releases) > 0:
        first_release_after_grasp_frame = int(post_grasp_releases[0])
        first_release_after_grasp_time  = float(times[first_release_after_grasp_frame])
    else:
        first_release_after_grasp_frame = N - 1
        first_release_after_grasp_time  = float(times[-1])

    # Longest consecutive closed run
    max_run = 0
    cur_run = 0
    for c in closed_mask:
        if c:
            cur_run += 1
            max_run = max(max_run, cur_run)
        else:
            cur_run = 0

    final_gripper_closed = bool(closed_mask[-1])

    # ── Z profile ─────────────────────────────────────────────────────────────
    z_start = float(z[0])
    z_end   = float(z[-1])
    z_rise  = float(np.max(z) - z_start)
    peak_z  = float(np.max(z))
    z_delta = z_end - z_start

    pct_idxs = [0, int(N*0.25), int(N*0.5), int(N*0.75), N-1]
    z_pct    = [round(float(z[i]), 4) for i in pct_idxs]

    # Z shape description
    if z_pct[2] > z_pct[0] + 0.03 and z_pct[4] < z_pct[2]:
        z_shape = "RISE-then-FALL (inverted V — arm went up and came back down)"
    elif z_pct[4] > z_pct[0] + 0.03:
        z_shape = "RISING overall (arm ended higher than it started)"
    elif z_pct[4] < z_pct[0] - 0.03:
        z_shape = "FALLING overall (arm ended lower than it started)"
    else:
        z_shape = "FLAT (minimal vertical change throughout)"

    # Z before grasp
    pre_grasp_z = z[:first_close_idx] if first_close_idx > 0 else z[:1]
    if len(pre_grasp_z) > 1:
        pre_trend = float(pre_grasp_z[-1] - pre_grasp_z[0])
        z_before_grasp_trend = f"{'descending' if pre_trend < -0.01 else 'ascending' if pre_trend > 0.01 else 'flat'} ({pre_trend:+.4f}m)"
    else:
        z_before_grasp_trend = "N/A"

    # Z rise AFTER grasp (did arm lift after closing gripper?)
    post_grasp_z = z[first_close_idx:]
    z_rise_after_grasp = float(np.max(post_grasp_z) - post_grasp_z[0]) if len(post_grasp_z) > 0 else 0.0

    # ── Ground Truth derivation (pick-and-place specific) ─────────────────────
    # Full success requires the complete cycle:
    #   1. Gripper closed (grasped)
    #   2. Z rose meaningfully after grasp (lifted)
    #   3. Gripper opened again after lift (placed/released)
    #   4. Closed gripper held for meaningful duration (not just a twitch)
    has_grasp   = grasp_events >= MIN_GRASP_EVENTS and closed_frames >= MIN_CLOSED_FRAMES
    has_lift    = z_rise_after_grasp >= Z_LIFT_MIN
    has_release = len(post_grasp_releases) > 0

    if has_grasp and has_lift and has_release:
        gt_label    = "success"
        gt_reasoning = (
            f"Full pick-and-place cycle detected: "
            f"grasp at t={first_close_time:.2f}s (closed for {max_run} frames), "
            f"z rose {z_rise_after_grasp:.3f}m after grasp, "
            f"released at t={first_release_after_grasp_time:.2f}s"
        )
    elif not has_grasp:
        gt_label    = "failure"
        gt_reasoning = (
            f"No stable grasp: grasp_events={grasp_events}, "
            f"max closed run={max_run} frames (need >={MIN_CLOSED_FRAMES})"
        )
    elif not has_lift:
        gt_label    = "failure"
        gt_reasoning = (
            f"Grasped but did not lift: z only rose {z_rise_after_grasp:.4f}m "
            f"after grasp (need >={Z_LIFT_MIN}m)"
        )
    elif not has_release:
        gt_label    = "failure"
        gt_reasoning = (
            f"Grasped and lifted but never released: "
            f"no open event detected after grasp at t={first_close_time:.2f}s"
        )
    else:
        gt_label    = "failure"
        gt_reasoning = "Incomplete pick-and-place cycle"

    return {
        "total_frames":                     N,
        "duration_sec":                     float(times[-1]),
        "pos_start":                        [round(float(x[0]),4), round(float(y[0]),4), round(float(z[0]),4)],
        "pos_end":                          [round(float(x[-1]),4), round(float(y[-1]),4), round(float(z[-1]),4)],
        "peak_z":                           peak_z,
        "z_rise":                           z_rise,
        "z_delta":                          z_delta,
        "z_start":                          z_start,
        "z_end":                            z_end,
        "z_pct":                            z_pct,
        "z_shape":                          z_shape,
        "z_before_grasp_trend":             z_before_grasp_trend,
        "z_rise_after_grasp":               z_rise_after_grasp,
        "total_xy_dist":                    float(np.sum(xy_dist)),
        "total_path_len":                   float(np.sum(step_dist)),
        "mean_vel":                         mean_vel,
        "max_vel":                          max_vel,
        "vel_std":                          vel_std,
        "abrupt_count":                     abrupt,
        "motion_quality":                   motion_quality,
        "gripper_open_frames":              open_frames,
        "gripper_closed_frames":            closed_frames,
        "gripper_open_pct":                 open_pct,
        "gripper_closed_pct":               closed_pct,
        "grasp_events":                     grasp_events,
        "release_events":                   release_events,
        "first_close_frame":                first_close_idx,
        "first_close_time":                 first_close_time,
        "first_release_after_grasp_frame":  first_release_after_grasp_frame,
        "first_release_after_grasp_time":   first_release_after_grasp_time,
        "max_closed_run":                   max_run,
        "final_gripper_closed":             final_gripper_closed,
        "gt_label":                         gt_label,
        "gt_reasoning":                     gt_reasoning,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VIDEO HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def download_video(ep_idx: int, view: str) -> Path:
    rel = f"videos/chunk-000/{view}/episode_{ep_idx:06d}.mp4"
    local = hf_hub_download(DATASET_REPO, rel, repo_type="dataset", token=HF_TOKEN)
    return Path(local)


def sample_frames(video_path: Path, num_frames: int) -> list:
    import av
    all_frames = []
    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            all_frames.append(frame.to_image())
    if not all_frames:
        raise RuntimeError(f"No frames decoded from {video_path}")
    n = len(all_frames)
    indices = [int(i * n / num_frames) for i in range(num_frames)]
    return [all_frames[min(i, n - 1)] for i in indices]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
def load_model_and_processor():
    print(f"\n📥  Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        _attn_implementation="eager",
    )
    model.eval()
    print("✅  Model ready\n")
    return model, processor


def build_messages(overhead_frames: list, wrist_frames: list, user_prompt: str) -> list:
    image_blocks = ([{"type": "image", "image": f} for f in overhead_frames] +
                    [{"type": "image", "image": f} for f in wrist_frames])
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": image_blocks + [{"type": "text", "text": user_prompt}]},
    ]


def run_inference(model, processor, messages: list) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [b["image"] for msg in messages for b in msg["content"] if b.get("type") == "image"]
    inputs = processor(text=text, images=images or None, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def parse_response(raw: str, video_id: str) -> dict:
    # Strip markdown fences if present
    clean = re.sub(r"^```[a-zA-Z]*\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean.strip())

    parsed = None
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        pass

    if parsed is None:
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
            except json.JSONDecodeError:
                pass

    if parsed is None:
        parsed = {
            "video_id":      video_id,
            "task_stages":   ["unknown"],
            "final_result":  "unknown",
            "reasoning":     raw,
            "confidence":    0.0,
            "parse_warning": "Model did not return valid JSON — raw text stored in reasoning",
        }

    parsed["video_id"] = video_id
    parsed.setdefault("final_result", "unknown")
    parsed.setdefault("confidence",   0.0)
    parsed.setdefault("task_stages",  [])
    parsed.setdefault("reasoning",    "")

    # Clean up: remove failure_reason if success
    if str(parsed.get("final_result", "")).lower() == "success":
        parsed.pop("failure_reason", None)

    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_all(predictions: list) -> dict:
    total   = len(predictions)
    correct = 0
    tp = tn = fp = fn = 0
    per_ep  = []

    for p in predictions:
        gt   = p.get("ground_truth_label", "unknown").lower().strip()
        pred = str(p.get("final_result", "unknown")).lower().strip()
        ok   = (gt == pred)
        if ok:
            correct += 1
        if   gt == "success" and pred == "success": tp += 1
        elif gt == "failure" and pred == "failure": tn += 1
        elif gt == "failure" and pred == "success": fp += 1
        elif gt == "success" and pred == "failure": fn += 1

        per_ep.append({
            "video_id":        p["video_id"],
            "gt_label":        gt,
            "predicted_label": pred,
            "correct":         ok,
            "confidence":      p.get("confidence", 0.0),
            "gt_reasoning":    p.get("gt_reasoning", ""),
            "task_stages":     p.get("task_stages", []),
        })

    acc  = correct / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "summary": {
            "total_episodes": total,
            "correct":        correct,
            "accuracy":       round(acc,  4),
            "precision":      round(prec, 4),
            "recall":         round(rec,  4),
            "f1_score":       round(f1,   4),
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives":fp,
            "false_negatives":fn,
        },
        "per_episode": per_ep,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if HF_TOKEN == "YOUR_HF_TOKEN_HERE":
        raise SystemExit("❌  export HF_TOKEN=hf_xxxxxxxxxxxx")

    login(token=HF_TOKEN)
    print("✅  Logged in to HuggingFace Hub")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂  Output dir: {OUTPUT_DIR.resolve()}")

    model, processor = load_model_and_processor()

    all_preds = []
    errors    = []

    for ep_idx in range(NUM_EPISODES):
        video_id = f"episode_{ep_idx:06d}.mp4"
        print(f"\n{'='*62}")
        print(f"🎬  Episode {ep_idx:02d}/{NUM_EPISODES-1}  →  {video_id}")
        print(f"{'='*62}")

        try:
            # 1. Features from parquet
            print("   📊  Extracting features from parquet ...")
            feat = extract_features(ep_idx)
            print(f"   GT label : {feat['gt_label'].upper():8s} | {feat['gt_reasoning']}")
            print(f"   Z rise after grasp: {feat['z_rise_after_grasp']:.4f}m | "
                  f"max closed run: {feat['max_closed_run']} frames | "
                  f"release events: {feat['release_events']}")

            # 2. Download both camera videos
            print("   ⬇️   Downloading videos ...")
            overhead_path = download_video(ep_idx, "observation.image")
            wrist_path    = download_video(ep_idx, "observation.wrist_image")

            # 3. Sample frames
            overhead_frames = sample_frames(overhead_path, num_frames=8)
            wrist_frames    = sample_frames(wrist_path,    num_frames=8)
            print(f"   🖼️   {len(overhead_frames)} overhead + {len(wrist_frames)} wrist frames sampled")

            # 4. Build prompt + messages
            user_prompt = build_user_prompt(ep_idx, feat)
            messages    = build_messages(overhead_frames, wrist_frames, user_prompt)

            # 5. Cosmos inference
            print("   🤖  Running Cosmos-Reason2 inference ...")
            raw = run_inference(model, processor, messages)
            print(f"   📝  Raw output (first 500 chars):\n   {raw[:500]}\n")

            # 6. Parse
            result = parse_response(raw, video_id)
            result["ground_truth_label"] = feat["gt_label"]
            result["gt_reasoning"]       = feat["gt_reasoning"]
            result["extracted_features"] = feat

            # 7. Save JSON
            out_path = PRED_DIR / f"episode_{ep_idx:06d}.json"
            with open(out_path, "w") as fh:
                json.dump(result, fh, indent=2)
            print(f"   💾  Saved → {out_path}")

            pred  = str(result.get("final_result", "unknown")).lower()
            gt    = feat["gt_label"]
            icon  = "✅" if pred == gt else "❌"
            conf  = result.get("confidence", 0.0)
            print(f"   {icon}  GT={gt.upper():<8} PRED={pred.upper():<8} Conf={conf:.2f}")

            all_preds.append(result)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"❌  Error on {video_id}:\n{tb}")
            err = {
                "video_id":            video_id,
                "task_stages":         [],
                "final_result":        "error",
                "reasoning":           f"Pipeline error: {e}",
                "confidence":          0.0,
                "ground_truth_label":  "unknown",
                "traceback":           tb,
            }
            out_path = PRED_DIR / f"episode_{ep_idx:06d}.json"
            with open(out_path, "w") as fh:
                json.dump(err, fh, indent=2)
            errors.append({"video_id": video_id, "error": str(e)})
            all_preds.append(err)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("📊  EVALUATION REPORT")
    print(f"{'='*62}")

    valid = [p for p in all_preds if p.get("final_result") not in ("error", "unknown")]
    report = evaluate_all(valid)

    eval_path = OUTPUT_DIR / "evaluation_report.json"
    with open(eval_path, "w") as fh:
        json.dump(report, fh, indent=2)

    s = report["summary"]
    print(f"\n  Episodes evaluated : {s['total_episodes']}")
    print(f"  Correct            : {s['correct']}")
    print(f"  Accuracy           : {s['accuracy']*100:.1f}%")
    print(f"  Precision          : {s['precision']*100:.1f}%")
    print(f"  Recall             : {s['recall']*100:.1f}%")
    print(f"  F1 Score           : {s['f1_score']*100:.1f}%")
    print(f"  TP={s['true_positives']}  TN={s['true_negatives']}  "
          f"FP={s['false_positives']}  FN={s['false_negatives']}")

    print(f"\n  {'VIDEO':<28} {'GT':<10} {'PRED':<10} {'OK':<4} {'CONF':<6} STAGES")
    print(f"  {'─'*80}")
    for ep in report["per_episode"]:
        icon   = "✅" if ep["correct"] else "❌"
        stages = "→".join(ep.get("task_stages", []))
        print(f"  {icon} {ep['video_id']:<26} {ep['gt_label'].upper():<10} "
              f"{ep['predicted_label'].upper():<10} "
              f"{str(ep['correct']):<4} {ep['confidence']:.2f}  {stages}")

    if errors:
        print(f"\n  ⚠️  {len(errors)} episodes errored (excluded from eval)")

    print(f"\n{'='*62}")
    print(f"✅  Pipeline done.")
    print(f"   Predictions → {PRED_DIR.resolve()}/")
    print(f"   Evaluation  → {eval_path.resolve()}")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()