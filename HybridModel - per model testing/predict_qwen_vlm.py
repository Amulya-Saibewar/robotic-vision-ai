"""
Qwen2.5-VL-7B as an alternative reasoning VLM.
Same prompt structure as Cosmos but uses a general-purpose VLM.
No physics-specific training — tests whether a generic VLM can do the job.

Requirements:
  pip install "transformers>=4.45" qwen-vl-utils torchcodec
"""
import json
import re
import textwrap
import warnings
from pathlib import Path

import torch
import transformers

SAMPLES  = json.load(open("../robot-task-monitoring/data/samples.json"))
MOTION_DIR = Path("../robot-task-monitoring/outputs/motion")
DEPTH_DIR  = Path("../robot-task-monitoring/outputs/depth")
OUT_FILE = Path("predictions_qwen_vlm.jsonl")

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"  # change to 2B if GPU limited

# --- Load model ---
print(f"Loading {MODEL_NAME}...")
processor = transformers.AutoProcessor.from_pretrained(MODEL_NAME)
model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
)
model.eval()
print("✓ Model loaded")


SCHEMA_BLOCK = textwrap.dedent("""
    OUTPUT SCHEMA (strict JSON only, no markdown, no extra text):
    {
      "video_id": "<str>",
      "stages": [{"name": "<approach|align|grasp|lift|move|place|idle>",
                  "start_s": <float>, "end_s": <float>}],
      "outcome": "<success|failure>",
      "failure_cause": "<str describing the failure, or none>",
      "confidence": <float 0-1>,
      "evidence": {
        "raft_vertical_drop_at_s": <float>,
        "depth_gap_to_target_mm":  <float>,
        "explanation": "<str>"
      }
    }
""").strip()


def build_signal_summary(vid: str, motion: dict, depth: dict) -> str:
    clip_s = motion.get("n_frames", 121) / motion.get("fps", 30.0)
    return textwrap.dedent(f"""
        VIDEO SIGNAL SUMMARY (video_id={vid}, clip_length={clip_s:.1f}s)
        [RAFT] peak_drop={motion.get('peak_vertical_drop', 0):.2f}px at t={motion.get('peak_time_s', 0):.2f}s,
               lift_onset={motion.get('lift_onset_s')}, place_onset={motion.get('place_onset_s')}
        [Depth] min_dist={depth.get('min_gripper_to_object_distance', 0):.2f},
                gap_to_target={depth.get('depth_gap_to_target_mm', 0):.1f}mm
    """).strip()


def predict_with_qwen(vid: str, rgb_path: str, motion: dict, depth: dict) -> dict:
    abs_path = str(Path(rgb_path).resolve())
    signal_summary = build_signal_summary(vid, motion, depth)

    messages = [
        {"role": "system", "content": [{"type": "text", "text":
            "You are a robot-task analysis model. Watch the video and the numeric signals, "
            "then output ONLY valid JSON matching the schema. No extra text."}]},
        {"role": "user", "content": [
            {"type": "video", "video": abs_path, "fps": 4},
            {"type": "text", "text": f"{SCHEMA_BLOCK}\n\n{signal_summary}\n\nOutput the JSON now."},
        ]},
    ]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt", truncation=False, fps=4,
    )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=2048, do_sample=False)

    prompt_len = inputs["input_ids"].shape[1]
    text = processor.decode(out_ids[0, prompt_len:], skip_special_tokens=True)

    # Extract JSON
    blocks = re.findall(r"\{.*\}", text, re.DOTALL)
    if blocks:
        try:
            data = json.loads(blocks[-1])
            if "video_id" not in data:
                data["video_id"] = vid
            return data
        except json.JSONDecodeError:
            pass

    # Fallback
    return {
        "video_id": vid,
        "stages": [{"name": "idle", "start_s": 0.0, "end_s": 4.0}],
        "outcome": "failure",
        "failure_cause": "parse_error",
        "confidence": 0.0,
        "evidence": {
            "raft_vertical_drop_at_s": 0.0,
            "depth_gap_to_target_mm": 0.0,
            "explanation": f"[Qwen-VLM] Failed to parse model output. Raw: {text[:200]}"
        }
    }


# --- Run ---
predictions = []
for i, sample in enumerate(SAMPLES):
    vid = sample["video_id"]
    print(f"\n[{i+1}/{len(SAMPLES)}] {vid}")

    motion_file = MOTION_DIR / f"{vid}.json"
    depth_file  = DEPTH_DIR / f"{vid}.json"

    motion = json.load(open(motion_file)) if motion_file.exists() else {}
    depth  = json.load(open(depth_file))  if depth_file.exists() else {}

    try:
        pred = predict_with_qwen(vid, sample["rgb_path"], motion, depth)
        predictions.append(pred)
        print(f"  ✓ outcome={pred['outcome']}, cause={pred.get('failure_cause','?')}, "
              f"conf={pred.get('confidence', 0):.2f}")
    except Exception as e:
        print(f"  ✗ error: {e}")
        predictions.append({
            "video_id": vid, "stages": [], "outcome": "failure",
            "failure_cause": "model_error", "confidence": 0.0,
            "evidence": {"raft_vertical_drop_at_s": 0.0, "depth_gap_to_target_mm": 0.0,
                         "explanation": f"[Qwen-VLM] Error: {str(e)[:200]}"}
        })

OUT_FILE.write_text("\n".join(json.dumps(p) for p in predictions) + "\n")
print(f"\n→ Saved {len(predictions)} predictions to {OUT_FILE}")
