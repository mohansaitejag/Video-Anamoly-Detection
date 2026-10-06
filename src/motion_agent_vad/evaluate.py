"""Anomaly-detection evaluation against CUHK Avenue frame-level ground truth.

Computes, for a set of (predicted anomaly score per frame, ground-truth
binary label per frame) pairs pooled across test videos:
  * Precision / Recall / F1 at the configured score threshold
  * Precision / Recall / F1 at the best-F1 threshold found by sweeping
  * ROC-AUC (only computed if both classes are present in the pooled labels)

and compares two systems on the exact same videos:
  * baseline  : optical flow -> regional motion -> anomaly score
                (no detection/tracking at all)
  * proposed  : YOLO -> ByteTrack -> object motion, combined with regional
                motion, -> contextual motion deviation -> anomaly score

All numbers are computed from whatever evidence/ground-truth was actually
supplied -- nothing here is hard-coded or invented.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Metrics:
    n_frames: int
    n_positive: int
    n_negative: int
    threshold: float
    precision: float
    recall: float
    f1: float
    best_f1_threshold: Optional[float]
    best_f1: Optional[float]
    best_f1_precision: Optional[float]
    best_f1_recall: Optional[float]
    auc: Optional[float]


def _prf1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float):
    preds = (scores >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def evaluate_scores(
    scores: List[float], labels: List[int], fixed_threshold: float
) -> Metrics:
    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.int64)
    assert len(scores_arr) == len(labels_arr), "scores/labels length mismatch"

    precision, recall, f1 = _prf1_at_threshold(scores_arr, labels_arr, fixed_threshold)

    best_f1, best_thr, best_p, best_r = None, None, None, None
    candidate_thresholds = np.unique(scores_arr)
    if len(candidate_thresholds) > 0:
        best_f1 = -1.0
        for thr in candidate_thresholds:
            p, r, f = _prf1_at_threshold(scores_arr, labels_arr, float(thr))
            if f > best_f1:
                best_f1, best_thr, best_p, best_r = f, float(thr), p, r

    auc = None
    if len(np.unique(labels_arr)) == 2:
        from sklearn.metrics import roc_auc_score

        auc = float(roc_auc_score(labels_arr, scores_arr))

    return Metrics(
        n_frames=len(labels_arr),
        n_positive=int(np.sum(labels_arr == 1)),
        n_negative=int(np.sum(labels_arr == 0)),
        threshold=fixed_threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        best_f1_threshold=best_thr,
        best_f1=best_f1 if best_f1 is not None and best_f1 >= 0 else None,
        best_f1_precision=best_p,
        best_f1_recall=best_r,
        auc=auc,
    )


@dataclass
class ComparisonReport:
    per_video: Dict[str, Dict[str, Optional[dict]]]
    pooled: Dict[str, Optional[dict]]

    def to_json(self, path: str) -> None:
        path = str(Path(path).expanduser())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    def to_markdown(self) -> str:
        lines = ["# Evaluation report", "", "## Pooled (all evaluated test videos)", ""]
        lines.append("| System | Precision | Recall | F1 | Best-F1 | AUC |")
        lines.append("|---|---|---|---|---|---|")
        for system in ("baseline", "proposed"):
            m = self.pooled.get(system)
            if m is None:
                lines.append(f"| {system} | - | - | - | - | - |")
                continue
            auc_str = f"{m['auc']:.3f}" if m["auc"] is not None else "n/a"
            best_str = f"{m['best_f1']:.3f}" if m["best_f1"] is not None else "n/a"
            lines.append(
                f"| {system} | {m['precision']:.3f} | {m['recall']:.3f} | "
                f"{m['f1']:.3f} | {best_str} | {auc_str} |"
            )
        lines.append("")
        lines.append("## Per-video")
        for video_name, systems in self.per_video.items():
            lines.append(f"\n### {video_name}")
            lines.append("| System | Precision | Recall | F1 | AUC |")
            lines.append("|---|---|---|---|---|")
            for system, m in systems.items():
                if m is None:
                    lines.append(f"| {system} | - | - | - | - |")
                    continue
                auc_str = f"{m['auc']:.3f}" if m["auc"] is not None else "n/a"
                lines.append(f"| {system} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {auc_str} |")
        return "\n".join(lines)


def compare_systems(
    per_video_scores: Dict[str, Dict[str, List[float]]],
    per_video_labels: Dict[str, List[int]],
    fixed_threshold: float,
) -> ComparisonReport:
    """per_video_scores[video_name] = {"baseline": [...], "proposed": [...]}."""
    per_video_report: Dict[str, Dict[str, Optional[dict]]] = {}
    pooled_scores: Dict[str, List[float]] = {"baseline": [], "proposed": []}
    pooled_labels: List[int] = []

    for video_name, systems in per_video_scores.items():
        labels = per_video_labels.get(video_name)
        if labels is None:
            per_video_report[video_name] = {"baseline": None, "proposed": None}
            continue
        entry: Dict[str, Optional[dict]] = {}
        for system in ("baseline", "proposed"):
            scores = systems.get(system)
            if scores is None or len(scores) != len(labels):
                entry[system] = None
                continue
            m = evaluate_scores(scores, labels, fixed_threshold)
            entry[system] = asdict(m)
            pooled_scores[system].extend(scores)
        per_video_report[video_name] = entry
        pooled_labels.extend(labels)

    pooled_report: Dict[str, Optional[dict]] = {}
    for system in ("baseline", "proposed"):
        if pooled_scores[system] and len(pooled_scores[system]) == len(pooled_labels):
            pooled_report[system] = asdict(evaluate_scores(pooled_scores[system], pooled_labels, fixed_threshold))
        else:
            pooled_report[system] = None

    return ComparisonReport(per_video=per_video_report, pooled=pooled_report)
