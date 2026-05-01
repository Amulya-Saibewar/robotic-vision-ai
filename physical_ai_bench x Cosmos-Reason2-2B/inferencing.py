"""
Cosmos-Reason2-2B Inference on shi-labs/physical-ai-bench-conditional-generation
=================================================================================
- Streams the dataset to get the relative video paths  (e.g. videos/task_0000.mp4)
- Downloads each video file directly from the HuggingFace Hub repo
- Runs Cosmos-Reason2-2B frame-level reasoning
- Writes one JSON per video into cosmos_outputs/

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxx
    python3 inferencing.py
"""

import os
import json
import re
import traceback
import tempfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
HF_TOKEN        = os.getenv("HF_TOKEN", "")
DATASET_REPO    = "shi-labs/physical-ai-bench-conditional-generation"
MODEL_ID        = "nvidia/Cosmos-Reason2-2B"
SPLIT           = "PAIBenchTransfer"
OUTPUT_DIR      = Path("cosmos_outputs")
NUM_VIDEOS      = 10
NUM_FRAMES      = 16
MAX_NEW_TOKENS  = 512
# ──────────────────────────────────────────────────────────────────────────────

import torch
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import login, hf_hub_download


# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a physical-AI task analyst. You are given uniformly sampled frames \
from a short robotic or human manipulation video.
 
Your task:
1. Identify the sequential action stages visible (choose from: approach, grasp, lift, move, \
   place, drop, release, retreat, idle, regrasp, success — use only what applies).
2. Provide a detailed, comprehensive explanation of everything happening in the video, including:
   - The objects involved and their properties
   - The robot/hand movements and actions in detail
   - The spatial positioning and how it changes
   - The sequence and timing of operations
   - How each action flows into the next
   - The final outcome and whether the task succeeded or failed
 
Reply with ONLY a valid JSON object — no markdown fences, no extra text:
{
  "video_id": "<filename>",
  "task_stages": ["<stage1>", "<stage2>", ...],
  "reasoning": "<comprehensive detailed explanation of all actions, movements, object interactions, spatial changes, and the complete outcome occurring throughout the entire video>"
}"""

USER_PROMPT = "Analyse these frames and return the JSON."
# ──────────────────────────────────────────────────────────────────────────────


def hf_login(token: str) -> None:
    login(token=token)
    print("✅  Logged in to HuggingFace Hub")


# ── Model ─────────────────────────────────────────────────────────────────────
def load_model_and_processor():
    print(f"\n📥  Loading {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, token=HF_TOKEN, trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("✅  Model ready\n")
    return model, processor


# ── Video helpers ─────────────────────────────────────────────────────────────
def download_video(relative_path: str) -> Path:
    """
    Download a dataset file (e.g. 'videos/task_0000.mp4') from the HF Hub
    into a local temp directory and return the local Path.

    hf_hub_download caches automatically — repeated calls are instant.
    """
    local_path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=relative_path,          # exact path inside the repo
        repo_type="dataset",
        token=HF_TOKEN,
    )
    return Path(local_path)


def sample_frames(video_path: Path, num_frames: int = NUM_FRAMES):
    """Uniformly sample `num_frames` PIL images from a video file."""
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


# ── Inference ─────────────────────────────────────────────────────────────────
def build_messages(frames: list) -> list:
    image_blocks = [{"type": "image", "image": f} for f in frames]
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": image_blocks + [{"type": "text", "text": USER_PROMPT}]},
    ]


def run_inference(model, processor, messages: list) -> str:
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images = [
        b["image"]
        for msg in messages
        for b in msg["content"]
        if b.get("type") == "image"
    ]
    inputs = processor(text=text, images=images or None, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False
        )

    new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()


def parse_response(raw: str, video_id: str) -> dict:
    """Try JSON parse; fall back to regex extraction; last resort: store raw text."""
    # Direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Extract first {...} block
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Fallback
    return {
        "video_id": video_id,
        "task_stages": ["unknown"],
        "failure_reason": raw,
        "parse_warning": "Model did not return valid JSON — raw output stored above",
    }


# ── Per-video pipeline ────────────────────────────────────────────────────────
def process_one_video(model, processor, relative_path: str, video_id: str) -> dict:
    print(f"\n🎬  {video_id}  ({relative_path})")

    # 1. Download from Hub
    print("   ⬇️   Downloading from HuggingFace Hub ...")
    local_path = download_video(relative_path)
    print(f"   📦  Local path: {local_path}  ({local_path.stat().st_size:,} bytes)")

    # 2. Sample frames
    frames = sample_frames(local_path)
    print(f"   🖼️   Sampled {len(frames)} frames")

    # 3. Build prompt & infer
    messages = build_messages(frames)
    print("   🤖  Running Cosmos-Reason2 inference ...")
    raw = run_inference(model, processor, messages)
    print(f"   📝  Raw output:\n{raw}\n")

    # 4. Parse
    result = parse_response(raw, video_id)
    result["video_id"] = video_id          # ensure correct id
    return result


def save_json(data: dict, video_id: str) -> Path:
    path = OUTPUT_DIR / f"{Path(video_id).stem}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"   💾  Saved → {path}")
    return path


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if HF_TOKEN == "YOUR_HF_TOKEN_HERE":
        raise SystemExit("❌  Set HF_TOKEN:  export HF_TOKEN=hf_xxxxxxxxxxxx")

    hf_login(HF_TOKEN)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂  Output dir : {OUTPUT_DIR.resolve()}")

    # ── Stream dataset to collect relative video paths ─────────────────────────
    print(f"\n📡  Streaming dataset split='{SPLIT}' ...")
    ds = load_dataset(
        DATASET_REPO,
        split=SPLIT,
        streaming=True,
        token=HF_TOKEN,
    )

    # Collect first NUM_VIDEOS examples
    examples = []
    for ex in ds:
        examples.append(ex)
        if len(examples) >= NUM_VIDEOS:
            break
    print(f"   Collected {len(examples)} examples")

    # Show schema from first example (helpful for debugging)
    if examples:
        print("\n   Dataset fields:")
        for k, v in examples[0].items():
            print(f"      '{k}': {type(v).__name__} = {str(v)[:120]}")

    # ── Determine the video field ──────────────────────────────────────────────
    video_field = None
    for candidate in ["video", "video_path", "mp4", "file", "clip", "path"]:
        if candidate in examples[0]:
            video_field = candidate
            break
    if video_field is None:
        # last resort: any string ending in .mp4
        for k, v in examples[0].items():
            if isinstance(v, str) and v.endswith(".mp4"):
                video_field = k
                break
    if video_field is None:
        raise SystemExit("❌  Could not detect a video field in the dataset. "
                         "Check the field printout above and set video_field manually.")

    print(f"\n   ✅  Video field: '{video_field}'\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model_and_processor()

    # ── Process each video ────────────────────────────────────────────────────
    results = []
    errors  = []

    for idx, example in enumerate(examples):
        relative_path = example[video_field]          # e.g. "videos/task_0000.mp4"
        video_id      = Path(relative_path).name      # e.g. "task_0000.mp4"

        try:
            result = process_one_video(model, processor, relative_path, video_id)
            save_json(result, video_id)
            results.append({"video_id": video_id, "status": "success"})

        except Exception as e:
            tb = traceback.format_exc()
            print(f"❌  Error on {video_id}:\n{tb}")
            err_result = {
                "video_id": video_id,
                "task_stages": [],
                "failure_reason": f"Processing error: {e}",
                "traceback": tb,
            }
            save_json(err_result, video_id)
            errors.append({"video_id": video_id, "error": str(e)})
            results.append({"video_id": video_id, "status": "error"})

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"results": results, "errors": errors}, f, indent=2)

    success = sum(1 for r in results if r["status"] == "success")
    print("\n" + "=" * 60)
    print(f"✅  Done.  {success} / {len(results)} videos succeeded.")
    if errors:
        print(f"⚠️   {len(errors)} errors logged in {summary_path}")
    print(f"📂  All outputs: {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()