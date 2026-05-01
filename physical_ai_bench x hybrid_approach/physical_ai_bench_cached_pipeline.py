"""
Physical-AI Bench Cached Pipeline v3 — FIXED
=============================================
Dataset : shi-labs/physical-ai-bench-conditional-generation
Models  :
  • depth-anything/Depth-Anything-V2-Large-hf   (Stage 4 — depth estimation)
  • facebook/vjepa2-vitl-fpc64-256              (Stage 6 — temporal understanding)
  • torchvision RAFT-Large                       (Stage 5 — optical flow / motion)
  • nvidia/Cosmos-Reason2-2B                    (Stage 8 — final reasoning)

All models are REQUIRED. No fallbacks. Run fails loud if any model cannot load.

Usage:
    export HF_TOKEN=hf_xxxxxxxxxxxx
    python3 pipeline.py [--video-id task_0000] [--force-recompute]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import pickle
import re
import time
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dotenv import load_dotenv

load_dotenv()

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from datasets import load_dataset
from huggingface_hub import hf_hub_download, login
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForDepthEstimation,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
HF_TOKEN         = os.getenv("HF_TOKEN", "")
DATASET_ID       = "shi-labs/physical-ai-bench-conditional-generation"
DATASET_SPLIT    = "PAIBenchTransfer"
CACHE_VERSION    = "v3"
TARGET_SIZE      = (224, 224)
TARGET_RGB_SIZE  = (224, 224)
TARGET_DEPTH_SIZE= (224, 224)

DEPTH_MODEL_ID   = "depth-anything/Depth-Anything-V2-Large-hf"
VJEPA2_MODEL_ID  = "facebook/vjepa2-vitl-fpc64-256"
COSMOS_MODEL_ID  = "nvidia/Cosmos-Reason2-2B"

PROJECT_ROOT = Path.cwd()
CACHE_ROOT   = PROJECT_ROOT / "cache"
OUTPUT_ROOT  = PROJECT_ROOT / "outputs"
HF_CACHE_ROOT= PROJECT_ROOT / "hf_cache"
LOG_ROOT     = PROJECT_ROOT / "logs"
for _d in (CACHE_ROOT, OUTPUT_ROOT, HF_CACHE_ROOT, LOG_ROOT):
    _d.mkdir(parents=True, exist_ok=True)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    else torch.float16
)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

FONT = ImageFont.load_default()

print(f"[INIT] DEVICE={DEVICE}  AMP_DTYPE={AMP_DTYPE}")
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def np_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):       return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)): return obj.item()
    if isinstance(obj, Path):             return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj).__name__}")


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=np_default)


# ══════════════════════════════════════════════════════════════════════════════
# LOGGER
# ══════════════════════════════════════════════════════════════════════════════
class PipelineLogger:
    def __init__(self, video_id: str, verbose: bool = True):
        self.video_id  = video_id
        self.verbose   = verbose
        self.records: List[Dict[str, Any]] = []
        self._t0: Optional[float] = None
        self._name: Optional[str] = None
        self._inputs: Dict = {}
        self._notes: List[str] = []

    @staticmethod
    def _desc(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return {"type": "ndarray", "shape": list(v.shape), "dtype": str(v.dtype)}
        if isinstance(v, torch.Tensor):
            return {"type": "tensor", "shape": list(v.shape)}
        if isinstance(v, Image.Image):
            return {"type": "PIL", "size": list(v.size)}
        if isinstance(v, (list, tuple)):
            return {"type": type(v).__name__, "len": len(v)}
        if isinstance(v, dict):
            return {"type": "dict", "keys": list(v.keys())[:8]}
        return v if isinstance(v, (int, float, bool, str, type(None))) else f"{type(v).__name__}(...)"

    def begin(self, name: str, **inputs: Any) -> None:
        self._t0, self._name, self._inputs, self._notes = time.time(), name, inputs, []
        if self.verbose:
            print(f"\n{'='*10} {name} {'='*10}")
            for k, v in inputs.items():
                print(f"  [INPUT]  {k} = {self._desc(v)}")

    def note(self, msg: str) -> None:
        self._notes.append(msg)
        if self.verbose:
            print(f"  [PROC ]  {msg}")

    def end(self, **outputs: Any) -> None:
        elapsed = time.time() - (self._t0 or time.time())
        if self.verbose:
            for k, v in outputs.items():
                print(f"  [OUTPUT] {k} = {self._desc(v)}")
            print(f"  [TIME ]  {elapsed:.3f}s")
        self.records.append({
            "stage": self._name,
            "inputs": {k: self._desc(v) for k, v in self._inputs.items()},
            "notes": list(self._notes),
            "outputs": {k: self._desc(v) for k, v in outputs.items()},
            "elapsed_s": round(elapsed, 4),
        })

    def write_to_disk(self) -> Path:
        path = LOG_ROOT / f"{self.video_id}.log.json"
        save_json(path, {"video_id": self.video_id, "stages": self.records})
        return path


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class SamplePaths:
    video_id: str
    rgb_path: str
    depth_path: str
    seg_pkl_path: str

@dataclass
class Stage1Output:
    rgb: np.ndarray
    depth: np.ndarray
    seg_masks: List[np.ndarray]
    seg_raw: Any

@dataclass
class Stage2Output:
    rgb: np.ndarray
    depth: np.ndarray
    seg: np.ndarray

@dataclass
class Stage3Output:
    object_mask: np.ndarray
    robot_mask: np.ndarray
    centroid: np.ndarray
    trajectory: np.ndarray
    velocity: np.ndarray
    interaction: np.ndarray

@dataclass
class Stage4Output:
    depth_maps: np.ndarray
    depth_curve: np.ndarray
    depth_diff: np.ndarray

@dataclass
class Stage5Output:
    motion_curve: np.ndarray

@dataclass
class Stage6Output:
    temporal_embeddings: np.ndarray
    video_embedding: np.ndarray

@dataclass
class Stage7Output:
    stage_hints: List[str]
    motion_pattern: str
    depth_pattern: str
    behavior_summary: str

@dataclass
class Stage8Output:
    task_stages: List[str]
    final_result: str
    failure_reason: str
    raw_text: str
    used_cosmos: bool = True
    cosmos_prompt: str = ""
    cosmos_canvas_size: Tuple[int, int] = (0, 0)


# ══════════════════════════════════════════════════════════════════════════════
# CACHE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _fingerprint(path_text: str) -> str:
    p = Path(path_text)
    if not p.exists():
        return f"missing:{path_text}"
    s = p.stat()
    d = hashlib.sha256()
    d.update(str(p.resolve()).encode())
    d.update(str(s.st_size).encode())
    d.update(str(int(s.st_mtime)).encode())
    return d.hexdigest()


def build_cache_key(sample: SamplePaths) -> str:
    payload = {
        "cache_version": CACHE_VERSION,
        "video_id": sample.video_id,
        "rgb": _fingerprint(sample.rgb_path),
        "depth": _fingerprint(sample.depth_path),
        "seg": _fingerprint(sample.seg_pkl_path),
        "depth_model": DEPTH_MODEL_ID,
        "vjepa2_model": VJEPA2_MODEL_ID,
        "cosmos_model": COSMOS_MODEL_ID,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_cache(video_id: str) -> Optional[Dict[str, Any]]:
    p = CACHE_ROOT / f"{video_id}.pkl.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def save_cache(video_id: str, payload: Dict[str, Any]) -> None:
    p = CACHE_ROOT / f"{video_id}.pkl.gz"
    with gzip.open(p, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET / IO HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def read_video_frames(path: str, gray: bool = False) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else frame
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded: {path}")
    return np.stack(frames, axis=0)


def load_segmentation_pkl(path: str) -> Tuple[List[np.ndarray], Any]:
    """
    PKL format:
      list of dicts, each with:
        'phrase': str
        'segmentation_mask_rle': {
            'data':       {'size': [T*H*W, 1], 'counts': bytes},
            'mask_shape': [T, H, W]
        }
    Decodes each phrase RLE → (T,H,W) binary mask, then builds a
    single (T,H,W) int32 label map (0=bg, 1=phrase0, 2=phrase1 ...).
    Returns list of T per-frame label arrays each shape (H,W).
    """
    from pycocotools import mask as coco_mask

    with open(path, "rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError(f"Unexpected PKL structure in {path}")

    first   = raw[0]["segmentation_mask_rle"]
    T, H, W = [int(x) for x in first["mask_shape"]]   # [T, H, W]

    label_map = np.zeros((T, H, W), dtype=np.int32)

    for idx, item in enumerate(raw):
        rle_data = item["segmentation_mask_rle"]["data"]
        flat_rle = {"size": rle_data["size"], "counts": rle_data["counts"]}
        decoded  = coco_mask.decode(flat_rle).reshape(T, H, W).astype(np.uint8)
        label_map[(decoded == 1) & (label_map == 0)] = idx + 1

    masks = [label_map[t] for t in range(T)]
    print(f"  [PKL] {len(raw)} phrases decoded → {T} frames shape=({H},{W}) labels=0..{len(raw)}")
    return masks, raw

def resize_stack(frames: np.ndarray, size: Tuple[int, int], interp: int) -> np.ndarray:
    H, W = size
    return np.stack([cv2.resize(f, (W, H), interpolation=interp) for f in frames])


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb.astype(np.float32) / 255.0, 0.0, 1.0)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    mx = float(np.max(depth))
    return depth / max(mx, 1.0)


def stack_seg_masks(masks: Sequence[np.ndarray]) -> np.ndarray:
    processed = []
    for m in masks:
        m = np.asarray(m)
        if m.ndim == 3:
            m = m[..., 0]
        if m.dtype == np.bool_:
            m = m.astype(np.uint8)
        elif np.issubdtype(m.dtype, np.floating):
            m = (m > 0.5).astype(np.uint8)
        else:
            m = m.astype(np.int32)
        processed.append(m)
    return np.stack(processed, axis=0).astype(np.int32)


def download_dataset_file(relative_path: str) -> str:
    """Download a file from the HF dataset repo and return local path."""
    path_text = relative_path.lstrip("./")
    local = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=path_text,
        token=HF_TOKEN,
        cache_dir=str(HF_CACHE_ROOT),
    )
    return local


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — Data Ingestion
# ══════════════════════════════════════════════════════════════════════════════
def stage_1_data_ingestion(
    rgb_path: str, depth_path: str, seg_pkl_path: str,
    logger: Optional[PipelineLogger] = None,
) -> Stage1Output:
    if logger:
        logger.begin("STAGE 1: data_ingestion",
                     rgb_path=rgb_path, depth_path=depth_path, seg_pkl_path=seg_pkl_path)
    rgb    = read_video_frames(rgb_path,   gray=False)
    depth  = read_video_frames(depth_path, gray=True)
    masks, raw = load_segmentation_pkl(seg_pkl_path)
    if logger:
        logger.note(f"RGB={rgb.shape}  depth={depth.shape}  masks={len(masks)}")
        logger.end(rgb=rgb, depth=depth, seg_masks=masks)
    return Stage1Output(rgb=rgb, depth=depth, seg_masks=masks, seg_raw=raw)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
def stage_2_preprocessing(
    rgb: np.ndarray, depth: np.ndarray, seg: np.ndarray,
    logger: Optional[PipelineLogger] = None,
) -> Stage2Output:
    if logger:
        logger.begin("STAGE 2: preprocessing", rgb=rgb, depth=depth, seg=seg)
    T = min(rgb.shape[0], depth.shape[0], seg.shape[0])
    rgb, depth, seg = rgb[:T], depth[:T], seg[:T]
    rgb_r   = normalize_rgb(resize_stack(rgb,   TARGET_RGB_SIZE,   cv2.INTER_LINEAR))
    depth_r = normalize_depth(resize_stack(depth, TARGET_DEPTH_SIZE, cv2.INTER_LINEAR))
    seg_r   = resize_stack(seg.astype(np.int32), TARGET_SIZE, cv2.INTER_NEAREST)
    if logger:
        logger.note(f"aligned T={T}, resized+normalised")
        logger.end(rgb=rgb_r, depth=depth_r, seg=seg_r)
    return Stage2Output(rgb=rgb_r.astype(np.float32),
                        depth=depth_r.astype(np.float32),
                        seg=seg_r.astype(np.int32))


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Segmentation Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════
def _label_stats(seg: np.ndarray) -> Dict[int, Dict]:
    T, H, W = seg.shape
    stats: Dict[int, Dict] = {}
    for t in range(T):
        frame = seg[t]
        for label in np.unique(frame):
            if int(label) == 0:
                continue
            mask = frame == label
            area = float(mask.sum())
            if area == 0:
                continue
            ys, xs = np.nonzero(mask)
            cx, cy = float(xs.mean()), float(ys.mean())
            border = float(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())
            bottom = cy / max(1.0, float(H))
            e = stats.setdefault(int(label), {
                "area_sum": 0.0, "frames": 0, "centroids": [],
                "border_sum": 0.0, "bottom_sum": 0.0,
            })
            e["area_sum"] += area
            e["frames"]   += 1
            e["centroids"].append(np.array([cx, cy], dtype=np.float32))
            e["border_sum"] += border
            e["bottom_sum"]  += bottom
    return stats


def _select_labels(seg: np.ndarray) -> Tuple[Optional[int], Optional[int]]:
    stats = _label_stats(seg)
    if not stats:
        return None, None
    T = float(seg.shape[0])
    obj_scores  = {}
    robot_scores = {}
    for label, info in stats.items():
        p = info["frames"] / max(1.0, T)
        a = info["area_sum"] / max(1.0, float(info["frames"]))
        b = info["border_sum"] / max(1.0, float(info["frames"]))
        bottom = info["bottom_sum"] / max(1.0, float(info["frames"]))
        obj_scores[label]   = a * p * (1.0 - 0.35 * b)
        robot_scores[label] = a * p * (0.5 + bottom + 0.5 * b)
    obj_label = max(obj_scores, key=obj_scores.get)
    robot_label = None
    candidates = {l: s for l, s in robot_scores.items() if l != obj_label}
    if candidates:
        robot_label = max(candidates, key=candidates.get)
    return obj_label, robot_label


def _centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def stage_3_segmentation_feature_extraction(
    seg: np.ndarray,
    logger: Optional[PipelineLogger] = None,
) -> Stage3Output:
    if logger:
        logger.begin("STAGE 3: segmentation_feature_extraction", seg=seg)
    T, H, W = seg.shape
    obj_label, robot_label = _select_labels(seg)
    obj_mask   = (seg == obj_label).astype(np.uint8)   if obj_label   is not None else (seg > 0).astype(np.uint8)
    robot_mask = (seg == robot_label).astype(np.uint8) if robot_label is not None else np.zeros_like(obj_mask)

    centroid    = np.zeros((T, 2), dtype=np.float32)
    trajectory  = np.zeros((T, 2), dtype=np.float32)
    velocity    = np.zeros((T, 2), dtype=np.float32)
    interaction = np.zeros((T,),   dtype=np.float32)
    fallback_c  = np.array([W * 0.5, H * 0.5], dtype=np.float32)
    robot_proxy = np.array([W * 0.5, H * 0.85], dtype=np.float32)

    for t in range(T):
        # object centroid
        oc = _centroid(obj_mask[t].astype(bool))
        if oc is None:
            oc = fallback_c.copy()
        centroid[t]   = oc
        trajectory[t] = oc
        if t > 0:
            velocity[t] = trajectory[t] - trajectory[t - 1]

        # robot centroid
        rc = _centroid(robot_mask[t].astype(bool))
        if rc is None:
            rc = robot_proxy.copy()

        obj_bool   = obj_mask[t].astype(bool)
        robot_bool = robot_mask[t].astype(bool)
        inter_area = float(np.logical_and(obj_bool, robot_bool).sum())
        union_area = float(np.logical_or(obj_bool,  robot_bool).sum())
        overlap    = inter_area / max(1.0, union_area)
        dist       = float(np.linalg.norm(oc - rc))
        prox       = math.exp(-dist / max(1.0, float(max(H, W))))
        interaction[t] = 0.5 * prox + 0.5 * overlap

    if logger:
        logger.note(f"obj_label={obj_label} robot_label={robot_label} "
                    f"mean_interaction={float(interaction.mean()):.3f}")
        logger.end(object_mask=obj_mask, robot_mask=robot_mask,
                   trajectory=trajectory, interaction=interaction)
    return Stage3Output(object_mask=obj_mask, robot_mask=robot_mask,
                        centroid=centroid, trajectory=trajectory,
                        velocity=velocity, interaction=interaction)

class DepthAnythingV2Extractor:
    """Depth-Anything-V2-Large for per-frame depth prediction."""

    def __init__(self):
        print(f"[DEPTH] Loading {DEPTH_MODEL_ID} ...")
        self.processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            DEPTH_MODEL_ID, torch_dtype=AMP_DTYPE, low_cpu_mem_usage=True,
        ).to(DEVICE)
        self.model.eval()
        print("[DEPTH] Ready")

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """rgb: (T, H, W, 3) float32 [0,1] → depth_maps: (T, H, W) float32"""
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"Expected (T,H,W,3), got {rgb.shape}")
        H, W = rgb.shape[1], rgb.shape[2]
        ctx = torch.autocast("cuda", dtype=AMP_DTYPE) if torch.cuda.is_available() else nullcontext()
        out_frames = []
        for frame in rgb:
            pil = Image.fromarray(np.clip(frame * 255, 0, 255).astype(np.uint8))
            inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v
                      for k, v in self.processor(images=pil, return_tensors="pt").items()}
            with ctx:
                outputs = self.model(**inputs)
            pred = outputs.predicted_depth  # (1, H', W') or (H', W')
            if pred.ndim == 2:
                pred = pred.unsqueeze(0).unsqueeze(0)
            elif pred.ndim == 3:
                pred = pred.unsqueeze(1)
            pred = F.interpolate(pred.float(), size=(H, W), mode="bilinear", align_corners=False)
            out_frames.append(pred.squeeze().cpu().numpy().astype(np.float32))
        return np.stack(out_frames, axis=0)


class RaftMotionExtractor:
    """RAFT-Large optical flow for per-frame motion magnitude."""

    def __init__(self):
        print("[RAFT] Loading RAFT-Large ...")
        weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=weights).to(DEVICE)
        self.model.eval()
        print("[RAFT] Ready")

    @torch.inference_mode()
    def flow(self, frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray:
        """frames: (H, W, 3) uint8 → flow: (H, W, 2) float32"""
        def to_tensor(f: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(f).permute(2, 0, 1).float().to(DEVICE).unsqueeze(0)
        ctx = torch.autocast("cuda", dtype=AMP_DTYPE) if torch.cuda.is_available() else nullcontext()
        with ctx:
            output = self.model(to_tensor(frame_a), to_tensor(frame_b))
        flow = output[-1] if isinstance(output, (list, tuple)) else output
        if flow.ndim == 4:
            flow = flow[0]
        H, W = frame_a.shape[:2]
        if tuple(flow.shape[-2:]) != (H, W):
            flow = F.interpolate(flow.unsqueeze(0), size=(H, W),
                                 mode="bilinear", align_corners=False)[0]
        return flow.permute(1, 2, 0).float().cpu().numpy().astype(np.float32)


class VJepa2TemporalExtractor:
    """V-JEPA 2 ViT-L for temporal video embeddings."""

    def __init__(self):
        print(f"[VJEPA2] Loading {VJEPA2_MODEL_ID} ...")
        self.processor = AutoProcessor.from_pretrained(VJEPA2_MODEL_ID, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            VJEPA2_MODEL_ID, torch_dtype=AMP_DTYPE,
            low_cpu_mem_usage=True, trust_remote_code=True,
        ).to(DEVICE)
        self.model.eval()
        print("[VJEPA2] Ready")

    def _pil_frames(self, rgb: np.ndarray) -> List[Image.Image]:
        return [Image.fromarray(np.clip(f * 255, 0, 255).astype(np.uint8)
                                if f.dtype != np.uint8 else f)
                for f in rgb]

    @torch.inference_mode()
    def _embed_clip(self, rgb: np.ndarray) -> np.ndarray:
        pil = self._pil_frames(rgb)
        # Try videos= first (native V-JEPA 2 API), fall back to images=
        try:
            inputs = self.processor(videos=pil, return_tensors="pt")
        except Exception:
            inputs = self.processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}
        ctx = torch.autocast("cuda", dtype=AMP_DTYPE) if torch.cuda.is_available() else nullcontext()
        with ctx:
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        for attr in ("pooler_output", "last_hidden_state", "hidden_states"):
            val = getattr(outputs, attr, None)
            if val is None:
                continue
            if attr == "hidden_states" and isinstance(val, (list, tuple)):
                val = val[-1]
            if isinstance(val, torch.Tensor):
                if val.ndim == 3:
                    val = val.mean(dim=1)
                return val.detach().float().mean(dim=0).cpu().numpy().astype(np.float32)
        raise RuntimeError("V-JEPA 2 did not expose any embedding tensor")

    @torch.inference_mode()
    def extract(self, rgb: np.ndarray, window: int = 16, stride: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        """rgb: (T,H,W,3) → (temporal_embeddings (T,D), video_embedding (D,))"""
        T = rgb.shape[0]
        rgb_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
        window = min(window, T)
        stride = max(1, stride)

        if T <= window:
            emb = self._embed_clip(rgb_u8)
            return np.repeat(emb[None], T, axis=0).astype(np.float32), emb.astype(np.float32)

        D = None
        temp_emb = None
        weights  = np.zeros((T, 1), dtype=np.float32)

        for start in range(0, T - window + 1, stride):
            end  = start + window
            emb  = self._embed_clip(rgb_u8[start:end])
            if D is None:
                D = emb.shape[0]
                temp_emb = np.zeros((T, D), dtype=np.float32)
            center = start + window // 2
            half   = max(1, window // 2)
            for fi in range(start, end):
                w = max(0.1, 1.0 - abs(fi - center) / float(half + 1))
                temp_emb[fi] += emb * w
                weights[fi]  += w

        # Fill any frames not covered
        missing = weights.squeeze(-1) <= 0
        if np.any(missing):
            fallback = self._embed_clip(rgb_u8)
            temp_emb[missing] = fallback
            weights[missing]  = 1.0

        temp_emb /= np.maximum(weights, 1e-6)
        return temp_emb.astype(np.float32), temp_emb.mean(axis=0).astype(np.float32)


@lru_cache(maxsize=1)
def get_depth_extractor() -> DepthAnythingV2Extractor:
    return DepthAnythingV2Extractor()

@lru_cache(maxsize=1)
def get_raft_extractor() -> RaftMotionExtractor:
    return RaftMotionExtractor()

@lru_cache(maxsize=1)
def get_vjepa2_extractor() -> VJepa2TemporalExtractor:
    return VJepa2TemporalExtractor()


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Hybrid Depth Feature Extraction (Depth-Anything-V2)
# ══════════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def stage_4_hybrid_depth_feature_extraction(
    rgb: np.ndarray, depth_video: np.ndarray, object_mask: np.ndarray,
    logger: Optional[PipelineLogger] = None,
) -> Stage4Output:
    if logger:
        logger.begin("STAGE 4: hybrid_depth_feature_extraction",
                     rgb=rgb, depth_video=depth_video, object_mask=object_mask)
    extractor = get_depth_extractor()
    if logger: logger.note(f"Running Depth-Anything-V2 on {rgb.shape[0]} frames ...")
    model_depth = normalize_depth(extractor.predict(rgb))
    dataset_depth = normalize_depth(depth_video)
    depth_final = 0.5 * model_depth + 0.5 * dataset_depth
    if logger: logger.note("fused: 0.5 * model_depth + 0.5 * dataset_depth")

    T = depth_final.shape[0]
    depth_curve = np.array([
        float(depth_final[t][object_mask[t].astype(bool)].mean())
        if object_mask[t].astype(bool).any() else float(depth_final[t].mean())
        for t in range(T)
    ], dtype=np.float32)
    depth_diff = np.zeros_like(depth_curve)
    if T > 1:
        depth_diff[1:] = depth_curve[1:] - depth_curve[:-1]

    if logger:
        logger.note(f"depth_curve range=[{depth_curve.min():.3f},{depth_curve.max():.3f}]")
        logger.end(depth_maps=depth_final, depth_curve=depth_curve, depth_diff=depth_diff)
    return Stage4Output(depth_maps=depth_final.astype(np.float32),
                        depth_curve=depth_curve, depth_diff=depth_diff)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Motion Feature Extraction (RAFT)
# ══════════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def stage_5_motion_feature_extraction(
    rgb: np.ndarray, object_mask: np.ndarray,
    logger: Optional[PipelineLogger] = None,
) -> Stage5Output:
    if logger:
        logger.begin("STAGE 5: motion_feature_extraction", rgb=rgb, object_mask=object_mask)
    T = rgb.shape[0]
    motion_curve = np.zeros((T,), dtype=np.float32)
    if T <= 1:
        if logger: logger.end(motion_curve=motion_curve)
        return Stage5Output(motion_curve=motion_curve)

    extractor = get_raft_extractor()
    rgb_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    if logger: logger.note(f"Running RAFT-Large optical flow on {T - 1} frame pairs ...")
    for t in range(T - 1):
        flow = extractor.flow(rgb_u8[t], rgb_u8[t + 1])
        mag  = np.linalg.norm(flow, axis=-1)
        mask = object_mask[t].astype(bool)
        motion_curve[t + 1] = float((mag[mask] if mask.any() else mag.reshape(-1)).mean())

    if logger:
        logger.note(f"mean_motion={motion_curve.mean():.3f}  peak={motion_curve.max():.3f}")
        logger.end(motion_curve=motion_curve)
    return Stage5Output(motion_curve=motion_curve)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — Temporal Understanding (V-JEPA 2)
# ══════════════════════════════════════════════════════════════════════════════
@torch.inference_mode()
def stage_6_temporal_understanding(
    rgb: np.ndarray,
    logger: Optional[PipelineLogger] = None,
) -> Stage6Output:
    if logger:
        logger.begin("STAGE 6: temporal_understanding", rgb=rgb)
    rgb_u8 = np.clip(rgb * 255, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    if logger: logger.note(f"Running V-JEPA 2 on clip shape={rgb_u8.shape} ...")
    temp_emb, vid_emb = get_vjepa2_extractor().extract(rgb_u8)
    if logger:
        logger.note(f"temporal_embeddings={temp_emb.shape}  video_embedding norm={float(np.linalg.norm(vid_emb)):.3f}")
        logger.end(temporal_embeddings=temp_emb, video_embedding=vid_emb)
    return Stage6Output(temporal_embeddings=temp_emb, video_embedding=vid_emb)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — Multimodal Fusion
# ══════════════════════════════════════════════════════════════════════════════
def _motion_pattern(mc: np.ndarray) -> str:
    mc = np.asarray(mc, dtype=np.float32)
    if mc.size == 0:              return "unknown"
    mean, std, peak = float(mc.mean()), float(mc.std()), float(mc.max())
    if peak < 1e-3:               return "static"
    if std / max(mean, 1e-6) > 1.5: return "irregular"
    if mean > 1.0 and std < 0.5 * mean: return "steady"
    if peak > 3.0 * max(mean, 1e-6):    return "burst"
    return "gradual"


def _depth_pattern(dc: np.ndarray, dd: np.ndarray) -> str:
    dc, dd = np.asarray(dc, dtype=np.float32), np.asarray(dd, dtype=np.float32)
    if dc.size == 0: return "unknown"
    change = float(dc[-1] - dc[0]) if dc.size > 1 else 0.0
    dstd   = float(dd.std()) if dd.size else 0.0
    if abs(change) < 0.02 and dstd < 0.02: return "stable"
    if change <= -0.05:                     return "rising"
    if change >= 0.05:                      return "falling"
    return "oscillating"


def _stage_hints(traj: np.ndarray, mc: np.ndarray, dc: np.ndarray,
                 dd: np.ndarray, inter: np.ndarray) -> List[str]:
    hints: List[str] = []
    mc, inter, dc, dd = (np.asarray(a, dtype=np.float32) for a in (mc, inter, dc, dd))
    if mc.size == 0: return ["idle"]
    mot_th = max(0.5, float(np.median(mc)) * 1.2)
    int_th = max(0.4, float(np.median(inter)) * 1.1)
    moving     = mc > mot_th
    in_contact = inter > int_th
    if inter.size > 1 and float(inter[-1] - inter[0]) > 0.05:
        hints.append("approach")
    if in_contact.any():
        hints.append("grasp")
        contact_idx = np.where(in_contact)[0]
        if contact_idx.size:
            s, e = int(contact_idx[0]), int(contact_idx[-1])
            if e > s and dc.size > e and float(dc[e] - dc[s]) < -0.03:
                hints.append("lift")
    if int(moving.sum()) >= max(2, int(0.2 * moving.size)):
        hints.append("move")
    if inter.size > 1:
        mi = int(np.argmax(inter))
        tail = inter[mi + 1:] if mi + 1 < inter.size else np.empty(0)
        if tail.size and float(tail.min()) < 0.3 and float(inter[mi]) > 0.5:
            hints.append("drop")
    return hints or ["idle"]


def stage_7_multimodal_fusion(
    trajectory: np.ndarray, depth_curve: np.ndarray,
    motion_curve: np.ndarray, interaction: np.ndarray,
    temporal_embedding: np.ndarray, depth_diff: Optional[np.ndarray] = None,
    logger: Optional[PipelineLogger] = None,
) -> Stage7Output:
    if logger:
        logger.begin("STAGE 7: multimodal_fusion",
                     trajectory=trajectory, depth_curve=depth_curve,
                     motion_curve=motion_curve, interaction=interaction)
    trajectory, depth_curve, motion_curve, interaction = (
        np.asarray(a, dtype=np.float32)
        for a in (trajectory, depth_curve, motion_curve, interaction)
    )
    if depth_diff is None:
        depth_diff = np.zeros_like(depth_curve)
        if depth_curve.size > 1:
            depth_diff[1:] = depth_curve[1:] - depth_curve[:-1]
    depth_diff = np.asarray(depth_diff, dtype=np.float32)

    mp  = _motion_pattern(motion_curve)
    dp  = _depth_pattern(depth_curve, depth_diff)
    hints = _stage_hints(trajectory, motion_curve, depth_curve, depth_diff, interaction)
    emb_norm = float(np.linalg.norm(np.asarray(temporal_embedding, dtype=np.float32).reshape(-1)))
    summary = (
        f"stages={'|'.join(hints)}; motion={mp}; depth={dp}; "
        f"traj_dx={float(trajectory[-1, 0] - trajectory[0, 0]):.1f}; "
        f"traj_dy={float(trajectory[-1, 1] - trajectory[0, 1]):.1f}; "
        f"peak_interaction={float(interaction.max()):.2f}; "
        f"embed_norm={emb_norm:.3f}"
    )
    if logger:
        logger.note(f"mp={mp} dp={dp} hints={hints}")
        logger.end(stage_hints=hints, motion_pattern=mp, depth_pattern=dp, behavior_summary=summary)
    return Stage7Output(stage_hints=hints, motion_pattern=mp,
                        depth_pattern=dp, behavior_summary=summary)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 — Cosmos-Reason2 Reasoning (REQUIRED, no fallback)
# ══════════════════════════════════════════════════════════════════════════════
COSMOS_SYSTEM_PROMPT = """You are a physical-AI analyst evaluating a robotic manipulation video.

You receive structured sensor signals computed from the video:
  • stage_hints     : action stages detected from motion, depth and segmentation signals
  • motion_pattern  : character of end-effector motion
  • depth_pattern   : object depth change over time
  • behavior_summary: compact feature string

TASK: Infer what happened in this robotic episode.

Watch the signals and reason about the causal sequence:
  - What stages did the robot go through?
  - Did it succeed or fail?
  - If it failed, what caused the failure?

Respond with ONLY a valid JSON object:
{
  "task_stages": ["<stage1>", "<stage2>", ...],
  "final_result": "success" or "failure",
  "failure_reason": "<precise cause if failure, else empty string>",
  "reasoning": "<brief description of what happened>"
}"""


def _render_canvas(fusion: Stage7Output, size: Tuple[int, int] = (1280, 800)) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    draw   = ImageDraw.Draw(canvas)
    lines  = [
        "Cosmos-Reason2 Input",
        f"stage_hints: {', '.join(fusion.stage_hints)}",
        f"motion_pattern: {fusion.motion_pattern}",
        f"depth_pattern: {fusion.depth_pattern}",
        f"behavior_summary: {fusion.behavior_summary}",
    ]
    y = 24
    for line in lines:
        draw.text((24, y), line, fill="black", font=FONT)
        y += 32
    return canvas


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S | re.I)
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try: return json.loads(m.group(0))
        except Exception: pass
    return None


class CosmosReasoner:
    """Cosmos-Reason2-2B wrapper. REQUIRED — raises if model cannot load."""

    def __init__(self):
        print(f"[COSMOS] Loading {COSMOS_MODEL_ID} ...")
        self.processor = AutoProcessor.from_pretrained(
            COSMOS_MODEL_ID, trust_remote_code=True, token=HF_TOKEN
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            COSMOS_MODEL_ID,
            torch_dtype=AMP_DTYPE,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            token=HF_TOKEN,
            _attn_implementation="eager",
        ).to(DEVICE)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            COSMOS_MODEL_ID, trust_remote_code=True, token=HF_TOKEN
        )
        print("[COSMOS] Ready")

    def _build_prompt(self, fusion: Stage7Output) -> str:
        return (
            f"{COSMOS_SYSTEM_PROMPT}\n\n"
            f"stage_hints: {fusion.stage_hints}\n"
            f"motion_pattern: {fusion.motion_pattern}\n"
            f"depth_pattern: {fusion.depth_pattern}\n"
            f"behavior_summary: {fusion.behavior_summary}\n\n"
            "Return JSON only."
        )

    @torch.inference_mode()
    def reason(self, fusion: Stage7Output,
               logger: Optional[PipelineLogger] = None) -> Stage8Output:
        prompt = self._build_prompt(fusion)
        canvas = _render_canvas(fusion)
        if logger:
            logger.begin("STAGE 8: cosmos_reasoning",
                         prompt_len=len(prompt), canvas_size=canvas.size)
            logger.note(f"PROMPT:\n{prompt}")

        messages = [
            {"role": "system", "content": [{"type": "text", "text": COSMOS_SYSTEM_PROMPT}]},
            {"role": "user",   "content": [
                {"type": "image", "image": canvas},
                {"type": "text",  "text": (
                    f"stage_hints: {fusion.stage_hints}\n"
                    f"motion_pattern: {fusion.motion_pattern}\n"
                    f"depth_pattern: {fusion.depth_pattern}\n"
                    f"behavior_summary: {fusion.behavior_summary}\n\n"
                    "Return JSON only."
                )},
            ]},
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [canvas]
        inputs = self.processor(text=text, images=images, return_tensors="pt")
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

        ctx = torch.autocast("cuda", dtype=AMP_DTYPE) if torch.cuda.is_available() else nullcontext()
        with ctx:
            generated = self.model.generate(
                **inputs, max_new_tokens=512, do_sample=False,
            )
        # Decode only new tokens
        new_ids  = generated[:, inputs["input_ids"].shape[1]:]
        raw_text = self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()
        if logger:
            logger.note(f"RAW OUTPUT: {raw_text[:600]}")

        parsed = _extract_json(raw_text)
        if parsed is None:
            raise RuntimeError(
                f"Cosmos did not return parseable JSON.\nRaw output:\n{raw_text}"
            )

        task_stages   = [str(s) for s in (parsed.get("task_stages") or []) if str(s)] or ["unknown"]
        final_result  = str(parsed.get("final_result", "unknown")).lower()
        failure_reason= str(parsed.get("failure_reason", "")).strip()
        reasoning     = str(parsed.get("reasoning", "")).strip()

        out = Stage8Output(
            task_stages=task_stages,
            final_result=final_result,
            failure_reason=failure_reason,
            raw_text=raw_text,
            used_cosmos=True,
            cosmos_prompt=prompt,
            cosmos_canvas_size=canvas.size,
        )
        if logger:
            logger.note(f"PARSED → final_result={final_result} stages={task_stages}")
            logger.end(task_stages=task_stages, final_result=final_result,
                       failure_reason=failure_reason, used_cosmos=True)
        return out


@lru_cache(maxsize=1)
def get_cosmos_reasoner() -> CosmosReasoner:
    return CosmosReasoner()


def stage_8_cosmos_reasoning(
    fusion: Stage7Output,
    logger: Optional[PipelineLogger] = None,
) -> Stage8Output:
    return get_cosmos_reasoner().reason(fusion, logger=logger)


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9 — Confidence Scoring
# ══════════════════════════════════════════════════════════════════════════════
def _embed_stability(emb: np.ndarray) -> float:
    emb = np.asarray(emb, dtype=np.float32)
    if emb.shape[0] <= 1: return 1.0
    diffs = np.diff(emb, axis=0)
    norms = np.linalg.norm(diffs, axis=1)
    denom = float(np.mean(np.linalg.norm(emb, axis=1)) + 1e-6)
    return float(np.clip(1.0 / (1.0 + float(norms.mean()) / denom), 0.0, 1.0))


def stage_9_confidence(
    s3: Stage3Output, s4: Stage4Output, s5: Stage5Output,
    s6: Stage6Output, s7: Stage7Output, s8: Stage8Output,
    logger: Optional[PipelineLogger] = None,
) -> float:
    if logger:
        logger.begin("STAGE 9: confidence_scoring")
    mc = np.asarray(s5.motion_curve, dtype=np.float32)
    dc = np.asarray(s4.depth_curve,  dtype=np.float32)
    inter = np.asarray(s3.interaction, dtype=np.float32)

    def stab(a: np.ndarray) -> float:
        return float(1.0 / (1.0 + np.std(a) / (np.mean(np.abs(a)) + 1e-6)))

    ms = stab(mc); ds = stab(dc); ins = stab(inter)
    ts = _embed_stability(s6.temporal_embeddings)

    hint_set    = {h.lower() for h in s7.stage_hints}
    pred_set    = {s.lower() for s in s8.task_stages}
    overlap     = len(hint_set & pred_set)
    align       = float(np.clip(0.6 + 0.4 * (overlap / max(1, len(pred_set))), 0.0, 1.0))
    if s8.final_result.lower() in ("success", "failure"):
        align = min(1.0, align + 0.05)

    signal_agr  = float(np.mean([ms, ds, ins]))
    confidence  = float(np.clip(0.35 * signal_agr + 0.25 * ts + 0.40 * align, 0.0, 0.99))
    if logger:
        logger.note(f"ms={ms:.3f} ds={ds:.3f} ins={ins:.3f} ts={ts:.3f} align={align:.3f}")
        logger.end(confidence=confidence)
    return confidence


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════
def assemble_output(video_id: str, s8: Stage8Output, confidence: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "video_id":      video_id,
        "task_stages":   s8.task_stages,
        "final_result":  s8.final_result,
        "confidence":    round(float(confidence), 4),
        "used_cosmos":   True,
        "reasoning":     s8.raw_text,
    }
    if s8.failure_reason:
        out["failure_reason"] = s8.failure_reason
    return out


def run_pipeline(
    video_id: str,
    rgb_path: str,
    depth_path: str,
    seg_pkl_path: str,
    force_recompute: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    logger = PipelineLogger(video_id=video_id, verbose=verbose)
    sample = SamplePaths(video_id=video_id, rgb_path=rgb_path,
                         depth_path=depth_path, seg_pkl_path=seg_pkl_path)
    cache_key   = build_cache_key(sample)
    cache_path  = CACHE_ROOT  / f"{video_id}.pkl.gz"
    output_path = OUTPUT_ROOT / f"{video_id}.json"

    # Cache hit
    if cache_path.exists() and not force_recompute:
        payload = load_cache(video_id)
        if payload and payload.get("cache_key") == cache_key:
            print(f"\n>>> CACHE HIT for {video_id}")
            if not output_path.exists():
                save_json(output_path, payload["final_output"])
            return payload["final_output"]

    print(f"\n>>> CACHE MISS for {video_id} — running full pipeline")

    # Run all stages
    s1 = stage_1_data_ingestion(rgb_path, depth_path, seg_pkl_path, logger)

    T = min(s1.rgb.shape[0], s1.depth.shape[0], len(s1.seg_masks))
    stacked_seg = stack_seg_masks(s1.seg_masks)[:T]

    s2 = stage_2_preprocessing(s1.rgb, s1.depth, stacked_seg, logger)
    s3 = stage_3_segmentation_feature_extraction(s2.seg, logger)
    s4 = stage_4_hybrid_depth_feature_extraction(s2.rgb, s2.depth, s3.object_mask, logger)
    s5 = stage_5_motion_feature_extraction(s2.rgb, s3.object_mask, logger)
    s6 = stage_6_temporal_understanding(s2.rgb, logger)
    s7 = stage_7_multimodal_fusion(
        trajectory=s3.trajectory, depth_curve=s4.depth_curve,
        motion_curve=s5.motion_curve, interaction=s3.interaction,
        temporal_embedding=s6.video_embedding, depth_diff=s4.depth_diff,
        logger=logger,
    )
    s8 = stage_8_cosmos_reasoning(s7, logger)

    if not s8.used_cosmos:
        raise RuntimeError("Stage 8 did not use Cosmos — this should never happen.")

    confidence   = stage_9_confidence(s3, s4, s5, s6, s7, s8, logger)
    final_output = assemble_output(video_id, s8, confidence)

    # Save cache, output, log
    save_cache(video_id, {
        "cache_key": cache_key,
        "final_output": final_output,
        "log_records": logger.records,
    })
    save_json(output_path, final_output)
    log_path = logger.write_to_disk()

    print(f"\n>>> FINAL OUTPUT: {final_output}")
    print(f">>> cache  → {cache_path}")
    print(f">>> output → {output_path}")
    print(f">>> log    → {log_path}")
    return final_output


def run_pipeline_from_dataset(
    video_id: str = "task_0000",
    force_recompute: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Download video/depth/seg files from HF Hub and run the pipeline."""
    print(f"[DATASET] Fetching files for video_id={video_id} ...")
    rgb_path     = download_dataset_file(f"videos/{video_id}.mp4")
    depth_path   = download_dataset_file(f"depth_vids/{video_id}.mp4")
    seg_pkl_path = download_dataset_file(f"sam2_pkls/{video_id}.pkl")
    return run_pipeline(
        video_id=video_id,
        rgb_path=rgb_path,
        depth_path=depth_path,
        seg_pkl_path=seg_pkl_path,
        force_recompute=force_recompute,
        verbose=verbose,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Physical-AI Bench Pipeline")
    parser.add_argument("--video-id", default="task_0000",
                        help="video_id to process (default: task_0000)")
    parser.add_argument("--force-recompute", action="store_true",
                        help="ignore cache and recompute all stages")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress per-stage verbose logging")
    args = parser.parse_args()


    login(token=HF_TOKEN)
    print("✅  Logged in to HuggingFace Hub")

    result = run_pipeline_from_dataset(
        video_id=args.video_id,
        force_recompute=args.force_recompute,
        verbose=not args.quiet,
    )
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(json.dumps(result, indent=2))
    print("="*60)