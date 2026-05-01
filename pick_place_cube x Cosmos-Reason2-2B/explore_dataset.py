"""
Dataset Explorer for theconstruct-ai/pick_place_cube
Run this first — paste the output back so we can build the full pipeline.

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxx
    python3 explore_dataset.py
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login, list_repo_files, hf_hub_download

load_dotenv()

HF_TOKEN     = os.getenv("HF_TOKEN", "")
DATASET_REPO = "theconstruct-ai/pick_place_cube"

def main():
    login(token=HF_TOKEN)
    print("✅  Logged in\n")

    # ── 1. List ALL files in the repo ─────────────────────────────────────────
    print("=" * 60)
    print("ALL FILES IN REPO:")
    print("=" * 60)
    all_files = list(list_repo_files(DATASET_REPO, repo_type="dataset", token=HF_TOKEN))
    for f in sorted(all_files):
        print(f"  {f}")

    print(f"\nTotal files: {len(all_files)}\n")

    # ── 2. Categorise by folder ───────────────────────────────────────────────
    folders = {}
    for f in all_files:
        folder = str(Path(f).parent)
        folders.setdefault(folder, []).append(Path(f).name)

    print("=" * 60)
    print("FILES BY FOLDER:")
    print("=" * 60)
    for folder, files in sorted(folders.items()):
        print(f"\n📁  {folder}/")
        for fname in sorted(files):
            print(f"      {fname}")

    # ── 3. Download and print every meta/*.json / *.jsonl file ────────────────
    meta_files = [f for f in all_files if "meta" in f.lower() and
                  (f.endswith(".json") or f.endswith(".jsonl"))]

    print("\n" + "=" * 60)
    print("META FILE CONTENTS:")
    print("=" * 60)

    for mf in sorted(meta_files):
        print(f"\n{'─'*50}")
        print(f"FILE: {mf}")
        print('─'*50)
        try:
            local = hf_hub_download(DATASET_REPO, mf, repo_type="dataset", token=HF_TOKEN)
            with open(local) as f:
                content = f.read()
            # Print first 3000 chars
            print(content[:3000])
            if len(content) > 3000:
                print(f"\n... (truncated, total {len(content)} chars)")
        except Exception as e:
            print(f"  ERROR reading: {e}")

    # ── 4. List video files ───────────────────────────────────────────────────
    video_files = [f for f in all_files if f.endswith(".mp4")]
    print("\n" + "=" * 60)
    print(f"VIDEO FILES ({len(video_files)} total):")
    print("=" * 60)
    for v in sorted(video_files)[:20]:   # first 20 only
        print(f"  {v}")
    if len(video_files) > 20:
        print(f"  ... and {len(video_files)-20} more")

    # ── 5. Check for parquet/data files ──────────────────────────────────────
    parquet_files = [f for f in all_files if f.endswith(".parquet")]
    print(f"\n{'='*60}")
    print(f"PARQUET/DATA FILES ({len(parquet_files)} total):")
    print("="*60)
    for p in sorted(parquet_files)[:10]:
        print(f"  {p}")

    if parquet_files:
        print("\n📊  Reading first parquet file columns + 2 rows...")
        try:
            import pandas as pd
            local = hf_hub_download(DATASET_REPO, parquet_files[0], repo_type="dataset", token=HF_TOKEN)
            df = pd.read_parquet(local)
            print(f"  Columns: {list(df.columns)}")
            print(f"  Shape: {df.shape}")
            print(f"  First 2 rows:\n{df.head(2).to_string()}")
        except Exception as e:
            print(f"  Could not read parquet: {e}")

    # ── 6. Try loading via HF datasets ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("LOADING VIA datasets LIBRARY:")
    print("="*60)
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_REPO, streaming=True, token=HF_TOKEN)
        print(f"  Splits available: {list(ds.keys())}")
        for split_name, split_ds in ds.items():
            print(f"\n  Split: '{split_name}'")
            ex = next(iter(split_ds))
            print(f"  Fields: {list(ex.keys())}")
            for k, v in ex.items():
                print(f"    '{k}': {type(v).__name__} = {str(v)[:150]}")
            break
    except Exception as e:
        print(f"  datasets load failed: {e}")

    print("\n" + "="*60)
    print("✅  Exploration complete. Paste this output to get the full pipeline.")
    print("="*60)

if __name__ == "__main__":
    main()