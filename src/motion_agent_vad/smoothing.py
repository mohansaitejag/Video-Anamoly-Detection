"""Exponential moving average temporal smoothing of anomaly scores.

Smoothing is kept per track_id (a person's own score history) and separately
a global/per-frame smoother is available for the region-only baseline system
which has no track identity.
"""
from __future__ import annotations

from typing import Dict, Optional


class EMASmoother:
    """smoothed_t = alpha * raw_t + (1 - alpha) * smoothed_{t-1}."""

    def __init__(self, alpha: float = 0.3):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self._state: Dict[int, float] = {}

    def update(self, key, raw_value: float) -> float:
        prev = self._state.get(key)
        smoothed = raw_value if prev is None else self.alpha * raw_value + (1 - self.alpha) * prev
        self._state[key] = smoothed
        return smoothed

    def reset(self, key: Optional[int] = None) -> None:
        if key is None:
            self._state.clear()
        else:
            self._state.pop(key, None)

    def value(self, key) -> Optional[float]:
        return self._state.get(key)
