"""Builds and persists the per-region "normal motion" baseline.

The baseline is computed ONLY from the normal (training) videos, as
required: for every one of the rows x cols regions we accumulate the mean
and standard deviation of the optical-flow magnitude observed across every
frame of every training video, using Welford's online algorithm so we never
need to hold all frames in memory at once.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MotionBaseline:
    rows: int
    cols: int
    mean: np.ndarray  # (rows, cols)
    std: np.ndarray  # (rows, cols)
    n_samples: np.ndarray  # (rows, cols) count of frames contributing, for diagnostics

    # -- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        path = str(Path(path).expanduser())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "rows": self.rows,
            "cols": self.cols,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "n_samples": self.n_samples.tolist(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "MotionBaseline":
        path = str(Path(path).expanduser())
        with open(path) as f:
            payload = json.load(f)
        return cls(
            rows=payload["rows"],
            cols=payload["cols"],
            mean=np.array(payload["mean"], dtype=np.float64),
            std=np.array(payload["std"], dtype=np.float64),
            n_samples=np.array(payload["n_samples"], dtype=np.int64),
        )

    def stats_for_region(self, row: int, col: int):
        return float(self.mean[row, col]), float(self.std[row, col])


class BaselineBuilder:
    """Accumulates per-region magnitude statistics frame by frame (Welford)."""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self._count = np.zeros((rows, cols), dtype=np.int64)
        self._mean = np.zeros((rows, cols), dtype=np.float64)
        self._m2 = np.zeros((rows, cols), dtype=np.float64)

    def update(self, mean_mag_grid: np.ndarray) -> None:
        """Feed one frame's per-region mean magnitude grid (rows, cols) into
        the running statistics. Each region contributes one sample per frame
        (its own spatial mean), which is the right granularity for comparing
        a tracked object's instantaneous motion against "typical" regional
        motion frame-to-frame.
        """
        self._count += 1
        delta = mean_mag_grid - self._mean
        self._mean += delta / self._count
        delta2 = mean_mag_grid - self._mean
        self._m2 += delta * delta2

    def finalize(self) -> MotionBaseline:
        variance = np.divide(
            self._m2,
            np.maximum(self._count - 1, 1),
            out=np.zeros_like(self._m2),
            where=self._count > 1,
        )
        std = np.sqrt(variance)
        return MotionBaseline(
            rows=self.rows,
            cols=self.cols,
            mean=self._mean.copy(),
            std=std,
            n_samples=self._count.copy(),
        )
