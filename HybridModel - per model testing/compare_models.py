"""
Compare predictions from all models side by side.
Generates:
  - comparison_table.csv  (per-sample, per-model outcomes)
  - comparison_summary.json (aggregate stats)
"""
import json
import csv
from pathlib import Path
from collections import Counter

SAMPLES = json.load(open("../robot-task-monitoring/data/samples.json"))
caption_map = {s["video_id"]: s.get("caption_text", "") for s in SAMPLES}

MODEL_FILES = {
    "RAFT":           "predictions_raft.jsonl",
    "Depth":          "predictions_depth.jsonl",
    "SAM2_Masks":     "predictions_masks.jsonl",
    "Cosmos":         "predictions_cosmos.jsonl",
    "Qwen_VLM":       "predictions_qwen_vlm.jsonl",
}

# Load all predictions
all_preds = {}
for model_name, filepath in MODEL_FILES.items():
    if not Path(filepath).exists():
        print(f"  ⚠ {filepath} not found — skipping {model_name}")
        continue
    preds = [json.loads(l) for l in open(filepath) if l.strip()]
    all_preds[model_name] = {p["video_id"]: p for p in preds}
    print(f"  ✓ {model_name}: {len(preds)} predictions loaded")

# Build comparison table
rows = []
for sample in SAMPLES:
    vid = sample["video_id"]
    row = {"video_id": vid, "caption": caption_map.get(vid, "")[:80]}
    for model_name in all_preds:
        pred = all_preds[model_name].get(vid, {})
        row[f"{model_name}_outcome"]    = pred.get("outcome", "N/A")
        row[f"{model_name}_cause"]      = pred.get("failure_cause", "N/A")
        row[f"{model_name}_confidence"] = pred.get("confidence", 0.0)
        row[f"{model_name}_n_stages"]   = len(pred.get("stages", []))
    rows.append(row)

# Write CSV
csv_path = Path("comparison_table.csv")
if rows:
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n→ Saved {csv_path}")

# Compute summary statistics
summary = {}
for model_name in all_preds:
    preds = list(all_preds[model_name].values())
    outcomes = Counter(p.get("outcome") for p in preds)
    causes   = Counter(p.get("failure_cause") for p in preds)
    confs    = [p.get("confidence", 0) for p in preds]
    n_stages = [len(p.get("stages", [])) for p in preds]

    summary[model_name] = {
        "n_predictions": len(preds),
        "success_count": outcomes.get("success", 0),
        "failure_count": outcomes.get("failure", 0),
        "success_rate":  round(outcomes.get("success", 0) / max(len(preds), 1), 3),
        "mean_confidence": round(sum(confs) / max(len(confs), 1), 3),
        "mean_n_stages":  round(sum(n_stages) / max(len(n_stages), 1), 1),
        "top_failure_causes": dict(causes.most_common(3)),
    }

summary_path = Path("comparison_summary.json")
summary_path.write_text(json.dumps(summary, indent=2))
print(f"→ Saved {summary_path}")

# Print comparison table
print(f"\n{'='*90}")
print(f"{'Model':<15} {'Samples':>8} {'Success':>8} {'Failure':>8} {'Rate':>8} {'Avg Conf':>9} {'Avg Stages':>11}")
print(f"{'-'*90}")
for model_name, stats in summary.items():
    print(f"{model_name:<15} {stats['n_predictions']:>8} {stats['success_count']:>8} "
          f"{stats['failure_count']:>8} {stats['success_rate']:>8.1%} "
          f"{stats['mean_confidence']:>9.2f} {stats['mean_n_stages']:>11.1f}")
print(f"{'='*90}")

# Agreement matrix
print(f"\n{'='*60}")
print("OUTCOME AGREEMENT MATRIX (% samples where models agree)")
print(f"{'='*60}")
model_names = list(all_preds.keys())
vids = [s["video_id"] for s in SAMPLES]
for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        if j <= i:
            continue
        agree = sum(
            1 for vid in vids
            if vid in all_preds.get(m1, {}) and vid in all_preds.get(m2, {})
            and all_preds[m1][vid].get("outcome") == all_preds[m2][vid].get("outcome")
        )
        total = sum(1 for vid in vids if vid in all_preds.get(m1, {}) and vid in all_preds.get(m2, {}))
        rate = agree / max(total, 1)
        print(f"  {m1} vs {m2}: {agree}/{total} ({rate:.0%})")
print(f"{'='*60}")
