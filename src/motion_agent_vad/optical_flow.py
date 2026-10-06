"""Farneback dense optical flow and 8x8 regional motion aggregation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class FlowParams:
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 15
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0

    @classmethod
    def from_config(cls, cfg) -> "FlowParams":
        return cls(
            pyr_scale=cfg.optical_flow.pyr_scale,
            levels=cfg.optical_flow.levels,
            winsize=cfg.optical_flow.winsize,
            iterations=cfg.optical_flow.iterations,
            poly_n=cfg.optical_flow.poly_n,
            poly_sigma=cfg.optical_flow.poly_sigma,
            flags=cfg.optical_flow.flags,
        )


def compute_farneback_flow(
    prev_gray: np.ndarray, curr_gray: np.ndarray, params: FlowParams
) -> np.ndarray:
    """Return a (H, W, 2) flow field (dx, dy per pixel)."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        params.pyr_scale,
        params.levels,
        params.winsize,
        params.iterations,
        params.poly_n,
        params.poly_sigma,
        params.flags,
    )
    return flow


def flow_to_magnitude_angle(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a (H, W, 2) flow field to magnitude and angle (radians, 0..2pi)."""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    return mag, ang


class RegionalMotionGrid:
    """Divides a frame into rows x cols regions and summarizes flow per region."""

    def __init__(self, rows: int = 8, cols: int = 8):
        self.rows = rows
        self.cols = cols

    def region_index_for_point(self, x: float, y: float, frame_w: int, frame_h: int) -> Tuple[int, int]:
        col = int(np.clip(x / max(frame_w, 1) * self.cols, 0, self.cols - 1))
        row = int(np.clip(y / max(frame_h, 1) * self.rows, 0, self.rows - 1))
        return row, col

    def cell_bounds(self, row: int, col: int, frame_w: int, frame_h: int):
        cell_h = frame_h / self.rows
        cell_w = frame_w / self.cols
        y0, y1 = int(row * cell_h), int((row + 1) * cell_h)
        x0, x1 = int(col * cell_w), int((col + 1) * cell_w)
        return x0, y0, x1, y1

    def compute_regional_stats(self, mag: np.ndarray, ang: np.ndarray):
        """Return per-cell mean magnitude, std magnitude, and mean (circular) angle.

        Shapes: mean_mag, std_mag, mean_ang all (rows, cols).
        """
        h, w = mag.shape
        mean_mag = np.zeros((self.rows, self.cols), dtype=np.float64)
        std_mag = np.zeros((self.rows, self.cols), dtype=np.float64)
        mean_ang = np.zeros((self.rows, self.cols), dtype=np.float64)

        for r in range(self.rows):
            for c in range(self.cols):
                x0, y0, x1, y1 = self.cell_bounds(r, c, w, h)
                cell_mag = mag[y0:y1, x0:x1]
                cell_ang = ang[y0:y1, x0:x1]
                if cell_mag.size == 0:
                    continue
                mean_mag[r, c] = float(np.mean(cell_mag))
                std_mag[r, c] = float(np.std(cell_mag))
                # circular mean of angle, weighted by magnitude so
                # near-static pixels don't dominate direction
                weights = cell_mag + 1e-8
                sin_sum = np.sum(np.sin(cell_ang) * weights)
                cos_sum = np.sum(np.cos(cell_ang) * weights)
                mean_ang[r, c] = float(np.arctan2(sin_sum, cos_sum) % (2 * np.pi))

        return mean_mag, std_mag, mean_ang


def magnitude_to_original_pixels_per_second(
    mag: np.ndarray, scale: float, fps: float
) -> np.ndarray:
    """Convert Farneback flow magnitude computed on a RESIZED frame (pixels
    per frame-step, at the resized resolution) into original-resolution
    pixels-per-second.

    This matters because object_motion.py derives a tracked person's speed
    in original-resolution pixels/second (bbox centers are rescaled back to
    the original frame by detection.py, and dt is real seconds). The
    regional-motion baseline (baseline.py) and the contextual z-scores in
    deviation.py compare a person's speed against the regional flow
    magnitude directly, so both MUST be in the same units, or the
    comparison is meaningless (comparing pixels/frame-at-reduced-resolution
    against pixels/second-at-full-resolution silently inflates z-scores by
    a large, resolution/fps-dependent constant factor). Always apply this
    conversion -- with the SAME `scale` and `fps` used by the video it came
    from -- before feeding magnitude into regional-stats aggregation,
    whether building the baseline or scoring a test video.
    """
    return mag * (fps / max(scale, 1e-6))


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def maybe_resize(frame: np.ndarray, longer_side: int | None):
    if not longer_side:
        return frame, 1.0
    h, w = frame.shape[:2]
    scale = longer_side / max(h, w)
    if scale >= 1.0:
        return frame, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale
