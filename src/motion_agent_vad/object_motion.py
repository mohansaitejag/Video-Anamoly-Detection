"""Per-track trajectory bookkeeping: position, speed, direction, acceleration.

ByteTrack (via Ultralytics) gives us persistent track IDs frame to frame.
This module keeps a short rolling history of each track's bounding-box
center and derives motion features from it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import numpy as np


@dataclass
class MotionSample:
    frame_idx: int
    t: float  # seconds
    cx: float
    cy: float


@dataclass
class MotionFeatures:
    speed: float  # pixels / second
    direction: float  # radians, 0..2pi, atan2 convention
    acceleration: Optional[float]  # pixels / second^2, None if not enough history


class TrackHistory:
    def __init__(self, history_length: int = 30):
        self.history_length = history_length
        self._tracks: Dict[int, Deque[MotionSample]] = {}

    def update(self, track_id: int, frame_idx: int, t: float, cx: float, cy: float) -> None:
        buf = self._tracks.setdefault(track_id, deque(maxlen=self.history_length))
        buf.append(MotionSample(frame_idx=frame_idx, t=t, cx=cx, cy=cy))

    def get(self, track_id: int) -> Deque[MotionSample]:
        return self._tracks.get(track_id, deque())

    def prune(self, active_ids) -> None:
        """Drop history for tracks that ByteTrack has dropped (left the scene)."""
        active = set(active_ids)
        for tid in list(self._tracks.keys()):
            if tid not in active:
                del self._tracks[tid]

    def features_for(self, track_id: int) -> MotionFeatures:
        buf = self._tracks.get(track_id)
        if not buf or len(buf) < 2:
            return MotionFeatures(speed=0.0, direction=0.0, acceleration=None)

        p1, p0 = buf[-1], buf[-2]
        dt = max(p1.t - p0.t, 1e-6)
        dx, dy = p1.cx - p0.cx, p1.cy - p0.cy
        speed = float(np.hypot(dx, dy) / dt)
        direction = float(np.arctan2(dy, dx) % (2 * np.pi))

        acceleration = None
        if len(buf) >= 3:
            p2 = buf[-3]
            dt_prev = max(p0.t - p2.t, 1e-6)
            dx_prev, dy_prev = p0.cx - p2.cx, p0.cy - p2.cy
            speed_prev = float(np.hypot(dx_prev, dy_prev) / dt_prev)
            acceleration = float((speed - speed_prev) / dt)

        return MotionFeatures(speed=speed, direction=direction, acceleration=acceleration)


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0
