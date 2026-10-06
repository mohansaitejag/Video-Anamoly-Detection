"""YOLO object detection + ByteTrack tracking (via Ultralytics `.track()`).

YOLO is used ONLY as an upstream detector to obtain person bounding boxes
and a persistent identity per person (through ByteTrack). It is deliberately
NOT used as the anomaly detector -- anomaly scoring happens later, in
deviation.py, from motion statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import resolve_device


@dataclass
class TrackedObject:
    track_id: int
    bbox: tuple  # (x1, y1, x2, y2) in original-frame pixel coords
    confidence: float
    cls: int


class Detector:
    """Thin wrapper around ultralytics YOLO(...).track(..., tracker='bytetrack.yaml')."""

    def __init__(self, cfg):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = resolve_device(cfg.detection.device)
        self.model = YOLO(cfg.detection.model)
        self.conf_threshold = cfg.detection.conf_threshold
        self.iou_threshold = cfg.detection.iou_threshold
        self.classes = list(cfg.detection.classes)
        self.tracker_config = cfg.tracking.tracker_config

    def reset(self) -> None:
        """Call between videos so ByteTrack doesn't carry IDs across clips."""
        # Ultralytics keeps tracker state on the predictor; the cleanest way
        # to reset it between videos is to drop the cached predictor so the
        # next .track() call reinitializes tracker state from scratch.
        self.model.predictor = None

    def track(self, frame: np.ndarray, scale: float = 1.0) -> List[TrackedObject]:
        """Run detection+tracking on one frame (already resized if scale != 1).

        Returns bounding boxes rescaled back to the ORIGINAL frame resolution
        (dividing by `scale`) so downstream region/grid math stays consistent
        with the un-resized frame.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            tracker=self.tracker_config,
            device=self.device,
            verbose=False,
        )

        objects: List[TrackedObject] = []
        if not results:
            return objects
        result = results[0]
        boxes = result.boxes
        if boxes is None or boxes.id is None:
            return objects

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        inv_scale = 1.0 / scale if scale else 1.0
        for i in range(len(ids)):
            x1, y1, x2, y2 = xyxy[i] * inv_scale
            objects.append(
                TrackedObject(
                    track_id=int(ids[i]),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(confs[i]),
                    cls=int(clss[i]),
                )
            )
        return objects
