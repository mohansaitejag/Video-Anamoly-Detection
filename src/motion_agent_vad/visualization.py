"""Drawing helpers used to build the annotated output video.

Everything here operates on a BGR uint8 frame (OpenCV convention) and
returns a new/annotated frame; nothing here affects the anomaly-scoring
math, it is purely for human inspection of the pipeline's behaviour.
"""
from __future__ import annotations

from typing import Dict, Iterable

import cv2
import numpy as np


def score_to_color(score: float, low: float = 0.0, high: float = 4.0) -> tuple:
    """Green (normal) -> Yellow -> Red (anomalous), BGR."""
    t = float(np.clip((score - low) / max(high - low, 1e-6), 0.0, 1.0))
    if t < 0.5:
        # green -> yellow
        u = t / 0.5
        b, g, r = 0, 255, int(255 * u)
    else:
        # yellow -> red
        u = (t - 0.5) / 0.5
        b, g, r = 0, int(255 * (1 - u)), 255
    return (b, g, r)


def draw_detections_and_scores(
    frame: np.ndarray,
    boxes: Iterable[dict],
) -> np.ndarray:
    """boxes: iterable of {track_id, bbox, score, is_anomalous}."""
    out = frame.copy()
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b["bbox"]]
        color = score_to_color(b.get("score", 0.0))
        thickness = 3 if b.get("is_anomalous") else 2
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"ID {b['track_id']} | s={b.get('score', 0.0):.2f}"
        if b.get("is_anomalous"):
            label += " ANOMALY"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
        )
    return out


def draw_flow_arrows(frame: np.ndarray, flow: np.ndarray, step: int = 16) -> np.ndarray:
    out = frame.copy()
    h, w = flow.shape[:2]
    for y in range(step // 2, h, step):
        for x in range(step // 2, w, step):
            dx, dy = flow[y, x]
            if np.hypot(dx, dy) < 0.5:
                continue
            end = (int(x + dx), int(y + dy))
            cv2.arrowedLine(out, (x, y), end, (0, 200, 255), 1, tipLength=0.35)
    return out


def draw_regional_grid(
    frame: np.ndarray,
    mean_mag_grid: np.ndarray,
    rows: int,
    cols: int,
    alpha: float = 0.35,
    vmax: float | None = None,
) -> np.ndarray:
    """vmax=None auto-scales to this grid's own 95th percentile (with a small
    floor) so the heatmap stays legible regardless of what unit system the
    motion magnitude happens to be in (raw flow, or the original-resolution
    pixels/second used by the pipeline)."""
    if vmax is None:
        vmax = max(float(np.percentile(mean_mag_grid, 95)), 1e-3)
    """Alpha-blended heatmap of current per-region motion magnitude, plus grid lines."""
    h, w = frame.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)
    cell_h, cell_w = h / rows, w / cols
    for r in range(rows):
        for c in range(cols):
            y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
            x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
            heat[y0:y1, x0:x1] = mean_mag_grid[r, c]

    heat_norm = np.clip(heat / max(vmax, 1e-6), 0, 1)
    heat_color = cv2.applyColorMap((heat_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(frame, 1 - alpha, heat_color, alpha, 0)

    for r in range(1, rows):
        y = int(r * cell_h)
        cv2.line(out, (0, y), (w, y), (255, 255, 255), 1, cv2.LINE_AA)
    for c in range(1, cols):
        x = int(c * cell_w)
        cv2.line(out, (x, 0), (x, h), (255, 255, 255), 1, cv2.LINE_AA)
    return out


def draw_trajectories(frame: np.ndarray, histories: Dict[int, list]) -> np.ndarray:
    """histories: {track_id: [(x, y), ...]} most-recent last."""
    out = frame.copy()
    for tid, pts in histories.items():
        if len(pts) < 2:
            continue
        color = _id_color(tid)
        for i in range(1, len(pts)):
            cv2.line(out, pts[i - 1], pts[i], color, 2, cv2.LINE_AA)
    return out


def _id_color(track_id: int) -> tuple:
    rng = np.random.default_rng(track_id * 9973 + 17)
    return tuple(int(v) for v in rng.integers(60, 255, size=3))


def draw_score_bar(frame: np.ndarray, raw_score: float, smoothed_score: float, threshold: float) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    bar_w, bar_h, x0, y0 = 220, 18, 10, 10

    def bar(y, value, vmax, color, label):
        cv2.rectangle(out, (x0, y), (x0 + bar_w, y + bar_h), (40, 40, 40), -1)
        filled = int(bar_w * np.clip(value / max(vmax, 1e-6), 0, 1))
        cv2.rectangle(out, (x0, y), (x0 + filled, y + bar_h), color, -1)
        cv2.putText(
            out,
            f"{label}: {value:.2f}",
            (x0 + 4, y + bar_h - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    vmax = max(threshold * 2, 4.0)
    bar(y0, raw_score, vmax, (120, 120, 120), "raw")
    bar(y0 + bar_h + 4, smoothed_score, vmax, score_to_color(smoothed_score, high=vmax), "smoothed")
    return out
