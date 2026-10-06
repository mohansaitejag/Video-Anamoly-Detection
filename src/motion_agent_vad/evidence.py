"""Structured, per-object evidence records, exported as CSV and JSON."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class EvidenceRecord:
    frame: int
    timestamp: float
    track_id: int
    bbox_x1: float
    bbox_y1: float
    bbox_x2: float
    bbox_y2: float
    region_row: int
    region_col: int
    speed: float
    direction: float
    acceleration: float
    regional_motion_mean: float
    regional_motion_current: float
    z_object: float
    z_region: float
    contextual_deviation: float
    anomaly_score_raw: float
    anomaly_score_smoothed: float
    confidence: float
    is_anomalous: bool


class EvidenceWriter:
    def __init__(self):
        self._records: List[EvidenceRecord] = []

    def add(self, record: EvidenceRecord) -> None:
        self._records.append(record)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[EvidenceRecord]:
        return self._records

    def to_csv(self, path: str) -> None:
        path = str(Path(path).expanduser())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not self._records:
            # still write a header-only file so downstream tools don't crash
            fieldnames = [f.name for f in EvidenceRecord.__dataclass_fields__.values()]
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(fieldnames)
            return
        fieldnames = list(asdict(self._records[0]).keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._records:
                writer.writerow(asdict(r))

    def to_json(self, path: str) -> None:
        path = str(Path(path).expanduser())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump([asdict(r) for r in self._records], f, indent=2)

    def frame_level_scores(self, n_frames: int) -> "list[float]":
        """Aggregate per-frame anomaly score as the MAX smoothed score among
        all objects present in that frame (frames with no detections -> 0).
        This is what evaluate.py compares against ground truth.
        """
        scores = [0.0] * n_frames
        for r in self._records:
            if 0 <= r.frame < n_frames:
                scores[r.frame] = max(scores[r.frame], r.anomaly_score_smoothed)
        return scores
