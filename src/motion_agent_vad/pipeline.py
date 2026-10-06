"""Orchestrates the full Motion Agent VAD pipeline, frame by frame:

Video -> preprocessing -> [YOLO+ByteTrack -> object motion] and
                           [Farneback flow -> 8x8 regional motion]
      -> contextual motion deviation -> anomaly score -> EMA smoothing
      -> evidence output (+ optional annotated video)

Two modes:
  * build_baseline(video_path, builder): accumulates regional motion stats
    into a BaselineBuilder. No detection/tracking is run (not needed, and
    keeps baseline-building fast).
  * run(video_path, baseline, ...): full inference pipeline on a test video,
    producing an EvidenceWriter (and optionally an annotated video file).
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .baseline import BaselineBuilder, MotionBaseline
from .detection import Detector
from .deviation import compute_contextual_deviation, regional_only_score
from .evidence import EvidenceRecord, EvidenceWriter
from .object_motion import TrackHistory, bbox_center
from .optical_flow import (
    FlowParams,
    RegionalMotionGrid,
    compute_farneback_flow,
    flow_to_magnitude_angle,
    magnitude_to_original_pixels_per_second,
    maybe_resize,
    to_grayscale,
)
from .smoothing import EMASmoother


def build_baseline(video_paths, cfg, progress_cb=None) -> MotionBaseline:
    grid = RegionalMotionGrid(cfg.grid.rows, cfg.grid.cols)
    flow_params = FlowParams.from_config(cfg)
    builder = BaselineBuilder(cfg.grid.rows, cfg.grid.cols)
    resize_to = cfg.optical_flow.resize_longer_side
    default_fps = cfg.object_motion.default_fps

    for video_path in video_paths:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or default_fps
        prev_gray = None
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_small, scale = maybe_resize(frame, resize_to)
            gray = to_grayscale(frame_small)
            if prev_gray is not None:
                flow = compute_farneback_flow(prev_gray, gray, flow_params)
                mag, _ = flow_to_magnitude_angle(flow)
                mag = magnitude_to_original_pixels_per_second(mag, scale, fps)
                mean_mag, _, _ = grid.compute_regional_stats(mag, np.zeros_like(mag))
                builder.update(mean_mag)
            prev_gray = gray
            frame_idx += 1
        cap.release()
        if progress_cb:
            progress_cb(str(video_path), frame_idx)

    return builder.finalize()


class MotionAgentPipeline:
    def __init__(self, cfg, baseline: Optional[MotionBaseline] = None, detector: Optional[Detector] = None):
        self.cfg = cfg
        self.baseline = baseline
        self.detector = detector or Detector(cfg)
        self.grid = RegionalMotionGrid(cfg.grid.rows, cfg.grid.cols)
        self.flow_params = FlowParams.from_config(cfg)
        self.resize_to = cfg.optical_flow.resize_longer_side

    def run(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> EvidenceWriter:
        if self.baseline is None:
            raise ValueError("A MotionBaseline must be built/loaded before running the pipeline.")

        from . import visualization as viz

        cfg = self.cfg
        eps = cfg.deviation.epsilon
        w_obj = cfg.deviation.weight_object_motion
        w_reg = cfg.deviation.weight_regional_motion
        max_z = cfg.deviation.get("max_z_score", 50.0)
        threshold = cfg.anomaly.score_threshold

        self.detector.reset()
        track_history = TrackHistory(cfg.object_motion.history_length)
        smoother = EMASmoother(cfg.smoothing.ema_alpha)
        baseline_smoother = EMASmoother(cfg.smoothing.ema_alpha)
        writer = EvidenceWriter()
        trail_points: dict[int, list] = {}

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or cfg.object_motion.default_fps
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        vw = None
        if output_video_path:
            Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*cfg.output.video_fourcc)
            vw = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_w, frame_h))

        prev_gray = None
        frame_idx = 0
        stride = max(1, cfg.output.frame_stride)

        # per-frame regional-only baseline-system score, kept for the
        # baseline-vs-proposed comparison in evaluate.py
        regional_only_scores: list[float] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            t = frame_idx / fps
            frame_small, scale = maybe_resize(frame, self.resize_to)
            gray = to_grayscale(frame_small)

            mean_mag = std_mag = mean_ang = None
            flow = None
            frame_regional_z = 0.0
            if prev_gray is not None:
                flow = compute_farneback_flow(prev_gray, gray, self.flow_params)
                mag, ang = flow_to_magnitude_angle(flow)
                mag = magnitude_to_original_pixels_per_second(mag, scale, fps)
                mean_mag, std_mag, mean_ang = self.grid.compute_regional_stats(mag, ang)

                # Detector-independent "baseline system" signal: how anomalous
                # is ANY region's current motion vs. its own baseline, over the
                # whole grid. This must not depend on whether a person was
                # detected, otherwise it isn't a fair optical-flow-only
                # comparator (a frame with zero detections would silently
                # score 0 even if the whole scene is behaving abnormally).
                for r in range(cfg.grid.rows):
                    for c in range(cfg.grid.cols):
                        mu_c, sigma_c = self.baseline.stats_for_region(r, c)
                        z_c = regional_only_score(float(mean_mag[r, c]), mu_c, sigma_c, eps, max_z)
                        frame_regional_z = max(frame_regional_z, z_c)

            objects = self.detector.track(frame_small, scale=scale)
            active_ids = [o.track_id for o in objects]
            track_history.prune(active_ids)

            render_boxes = []

            for obj in objects:
                cx, cy = bbox_center(obj.bbox)
                track_history.update(obj.track_id, frame_idx, t, cx, cy)
                motion = track_history.features_for(obj.track_id)

                if mean_mag is not None:
                    row, col = self.grid.region_index_for_point(cx, cy, frame_w, frame_h)
                    mu, sigma = self.baseline.stats_for_region(row, col)
                    region_now = float(mean_mag[row, col])
                    dev = compute_contextual_deviation(
                        object_speed=motion.speed,
                        region_current_mag=region_now,
                        region_mu=mu,
                        region_sigma=sigma,
                        eps=eps,
                        weight_object_motion=w_obj,
                        weight_regional_motion=w_reg,
                        max_z=max_z,
                    )
                else:
                    row, col, mu, region_now = -1, -1, 0.0, 0.0
                    dev = compute_contextual_deviation(0, 0, 0, 1, eps, w_obj, w_reg, max_z)

                smoothed = smoother.update(obj.track_id, dev.anomaly_score_raw)
                is_anomalous = smoothed >= threshold

                writer.add(
                    EvidenceRecord(
                        frame=frame_idx,
                        timestamp=round(t, 3),
                        track_id=obj.track_id,
                        bbox_x1=obj.bbox[0],
                        bbox_y1=obj.bbox[1],
                        bbox_x2=obj.bbox[2],
                        bbox_y2=obj.bbox[3],
                        region_row=row,
                        region_col=col,
                        speed=round(motion.speed, 3),
                        direction=round(motion.direction, 3),
                        acceleration=round(motion.acceleration, 3) if motion.acceleration is not None else 0.0,
                        regional_motion_mean=round(mu, 4),
                        regional_motion_current=round(region_now, 4),
                        z_object=round(dev.z_object, 4),
                        z_region=round(dev.z_region, 4),
                        contextual_deviation=round(dev.anomaly_score_raw, 4),
                        anomaly_score_raw=round(dev.anomaly_score_raw, 4),
                        anomaly_score_smoothed=round(smoothed, 4),
                        confidence=round(obj.confidence, 4),
                        is_anomalous=bool(is_anomalous),
                    )
                )

                trail = trail_points.setdefault(obj.track_id, [])
                trail.append((int(cx), int(cy)))
                if len(trail) > cfg.object_motion.history_length:
                    trail.pop(0)

                render_boxes.append(
                    {
                        "track_id": obj.track_id,
                        "bbox": obj.bbox,
                        "score": smoothed,
                        "is_anomalous": is_anomalous,
                    }
                )

            regional_only_scores.append(baseline_smoother.update("global", frame_regional_z))

            if vw is not None:
                out_frame = frame.copy()
                if flow is not None and scale != 1.0:
                    disp_flow = cv2.resize(flow, (frame_w, frame_h)) / max(scale, 1e-6)
                elif flow is not None:
                    disp_flow = flow
                else:
                    disp_flow = None

                if mean_mag is not None:
                    out_frame = viz.draw_regional_grid(out_frame, mean_mag, cfg.grid.rows, cfg.grid.cols)
                if disp_flow is not None:
                    out_frame = viz.draw_flow_arrows(out_frame, disp_flow)
                out_frame = viz.draw_trajectories(out_frame, trail_points)
                out_frame = viz.draw_detections_and_scores(out_frame, render_boxes)
                max_smoothed = max([b["score"] for b in render_boxes], default=0.0)
                out_frame = viz.draw_score_bar(out_frame, frame_regional_z, max_smoothed, threshold)
                vw.write(out_frame)

            prev_gray = gray
            frame_idx += 1

        cap.release()
        if vw is not None:
            vw.release()

        writer.regional_only_scores = regional_only_scores  # type: ignore[attr-defined]
        writer.n_frames_processed = frame_idx  # type: ignore[attr-defined]
        return writer
