"""CUHK Avenue dataset discovery and ground-truth loading.

The Avenue dataset is distributed with slightly different folder layouts
depending on where it's downloaded from, so this module searches a list of
candidate sub-folder names (from config) instead of assuming one fixed
layout. It also supports both of the common ground-truth formats:

  1. Per-video .mat pixel-mask volumes (e.g. ``1_label.mat`` containing a
     variable such as ``volLabel``), the format shipped in the official
     ``ground_truth_demo/testing_label_mask`` folder. A frame is labelled
     anomalous if any pixel in its mask is non-zero.
  2. A flat frame-level ground truth (e.g. a .txt/.npy of 0/1 per frame),
     in case the user supplies their own.

If no ground truth can be found for a video, functions return ``None`` and
callers are expected to handle that (e.g. skip that video during
evaluation) rather than fail silently with fabricated labels.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np


@dataclass
class VideoEntry:
    path: str
    name: str  # stem, used to match ground truth files, e.g. "01"


@dataclass
class AvenueDataset:
    root: str
    train_videos: List[VideoEntry] = field(default_factory=list)
    test_videos: List[VideoEntry] = field(default_factory=list)
    train_dir: Optional[str] = None
    test_dir: Optional[str] = None
    gt_dir: Optional[str] = None

    # -- discovery ----------------------------------------------------------
    @classmethod
    def discover(cls, cfg) -> "AvenueDataset":
        root = Path(os.path.expanduser(cfg.dataset.root))
        if not root.exists():
            raise FileNotFoundError(
                f"Dataset root does not exist: {root}\n"
                "Set dataset.root in your config (or pass --data-root) to the "
                "folder that contains the Avenue training/testing video folders."
            )

        exts = tuple(cfg.dataset.video_extensions)

        train_dir = _find_dir_with_videos(root, cfg.dataset.train_dir_candidates, exts)
        test_dir = _find_dir_with_videos(root, cfg.dataset.test_dir_candidates, exts)
        gt_dir = _find_existing_dir(root, cfg.dataset.gt_dir_candidates)

        ds = cls(root=str(root), train_dir=train_dir, test_dir=test_dir, gt_dir=gt_dir)
        if train_dir:
            ds.train_videos = _list_videos(train_dir, exts)
        if test_dir:
            ds.test_videos = _list_videos(test_dir, exts)
        return ds

    def summary(self) -> str:
        lines = [
            f"Dataset root : {self.root}",
            f"Training dir : {self.train_dir or '(not found)'} "
            f"({len(self.train_videos)} videos)",
            f"Testing dir  : {self.test_dir or '(not found)'} "
            f"({len(self.test_videos)} videos)",
            f"Ground truth : {self.gt_dir or '(not found)'}",
        ]
        return "\n".join(lines)


def _find_dir_with_videos(root: Path, candidates: List[str], exts) -> Optional[str]:
    for cand in candidates:
        d = root / cand
        if d.is_dir() and _list_videos(str(d), exts):
            return str(d)
    # Fallback: shallow search one level deep for any folder whose name
    # contains "train"/"test" and holds video files.
    for d in root.glob("*"):
        if d.is_dir() and _list_videos(str(d), exts):
            return str(d)
    return None


def _find_existing_dir(root: Path, candidates: List[str]) -> Optional[str]:
    for cand in candidates:
        d = root / cand
        if d.is_dir():
            return str(d)
    return None


def _list_videos(dir_path: str, exts) -> List[VideoEntry]:
    entries = []
    for ext in exts:
        for p in sorted(glob.glob(os.path.join(dir_path, f"*{ext}"))):
            entries.append(VideoEntry(path=p, name=Path(p).stem))
    entries.sort(key=lambda e: e.name)
    return entries


# -- ground truth ------------------------------------------------------------
def load_frame_level_ground_truth(
    video: VideoEntry, gt_dir: Optional[str], n_frames: int
) -> Optional[np.ndarray]:
    """Return a (n_frames,) binary array (1 = anomalous) for a test video, or None.

    Tries, in order:
      * ``{gt_dir}/{name}_label.mat`` with a pixel-mask volume variable
        (``volLabel``, ``labels``, or the first array-like variable found).
      * ``{gt_dir}/{name}.mat`` (same handling).
      * ``{gt_dir}/{name}.npy`` -- a plain (n_frames,) 0/1 array.
      * ``{gt_dir}/{name}.txt`` -- one 0/1 value per line, or a list of
        anomalous frame-index ranges "start-end" per line.
    """
    if not gt_dir:
        return None
    name = video.name

    for cand in (f"{name}_label.mat", f"{name}.mat"):
        mat_path = os.path.join(gt_dir, cand)
        if os.path.exists(mat_path):
            labels = _labels_from_mat(mat_path, n_frames)
            if labels is not None:
                return labels

    npy_path = os.path.join(gt_dir, f"{name}.npy")
    if os.path.exists(npy_path):
        arr = np.load(npy_path).astype(int).ravel()
        return _fit_length(arr, n_frames)

    txt_path = os.path.join(gt_dir, f"{name}.txt")
    if os.path.exists(txt_path):
        return _labels_from_txt(txt_path, n_frames)

    return None


def _labels_from_mat(mat_path: str, n_frames: int) -> Optional[np.ndarray]:
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scipy is required to read Avenue .mat ground truth") from exc

    try:
        mat = loadmat(mat_path)
    except Exception:
        return None

    candidate_keys = [k for k in mat.keys() if not k.startswith("__")]
    preferred = [k for k in candidate_keys if k.lower() in ("vollabel", "labels", "label", "l")]
    keys_to_try = preferred + [k for k in candidate_keys if k not in preferred]

    for key in keys_to_try:
        val = mat[key]
        labels = _array_to_frame_labels(val, n_frames)
        if labels is not None:
            return labels
    return None


def _array_to_frame_labels(val: np.ndarray, n_frames: int) -> Optional[np.ndarray]:
    val = np.asarray(val)
    if val.ndim == 0:
        return None

    # Common Avenue layout: object array of length n_frames, each a HxW mask.
    if val.dtype == object:
        flat = val.ravel()
        if len(flat) == 0:
            return None
        labels = np.zeros(len(flat), dtype=int)
        for i, cell in enumerate(flat):
            cell = np.asarray(cell)
            labels[i] = int(np.any(cell != 0))
        return _fit_length(labels, n_frames)

    # 3D numeric volume: (H, W, T) or (T, H, W).
    if val.ndim == 3:
        if val.shape[-1] == n_frames or (val.shape[-1] > val.shape[0] and val.shape[-1] > 1):
            # assume (H, W, T)
            per_frame = np.any(val.reshape(-1, val.shape[-1]) != 0, axis=0).astype(int)
        else:
            # assume (T, H, W)
            per_frame = np.any(val.reshape(val.shape[0], -1) != 0, axis=1).astype(int)
        return _fit_length(per_frame, n_frames)

    # Already frame-level (1D or Nx1/1xN).
    if val.ndim <= 2 and (val.ndim == 1 or 1 in val.shape):
        labels = (val.ravel() != 0).astype(int)
        return _fit_length(labels, n_frames)

    return None


def _fit_length(labels: np.ndarray, n_frames: int) -> np.ndarray:
    """Pad/truncate a label array to exactly n_frames (off-by-one guards)."""
    labels = np.asarray(labels).astype(int).ravel()
    if len(labels) == n_frames:
        return labels
    if len(labels) > n_frames:
        return labels[:n_frames]
    padded = np.zeros(n_frames, dtype=int)
    padded[: len(labels)] = labels
    return padded


def _labels_from_txt(txt_path: str, n_frames: int) -> np.ndarray:
    labels = np.zeros(n_frames, dtype=int)
    with open(txt_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if all(ln in ("0", "1") for ln in lines):
        vals = np.array([int(ln) for ln in lines])
        return _fit_length(vals, n_frames)
    # else treat as "start-end" ranges (1-indexed, inclusive), one per line
    for ln in lines:
        if "-" in ln:
            start, end = ln.split("-")
            start, end = int(start) - 1, int(end) - 1
            start = max(0, start)
            end = min(n_frames - 1, end)
            labels[start : end + 1] = 1
    return labels


def get_video_frame_count(video_path: str) -> int:
    import cv2

    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n
