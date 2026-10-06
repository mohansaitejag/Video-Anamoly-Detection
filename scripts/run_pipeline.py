#!/usr/bin/env python3
"""Run the full Motion Agent VAD pipeline on CUHK Avenue test video(s).

For each processed video this writes, under --output-dir:
    evidence/{name}.csv                per-object structured evidence
    evidence/{name}.json               same, as JSON
    scores/{name}_regional_baseline.json   per-frame score of the "baseline"
                                            (optical-flow-only) comparator,
                                            used later by scripts/evaluate.py
    videos/{name}_annotated.mp4        annotated video (unless --no-video)

Usage:
    python scripts/run_pipeline.py --data-root "~/Downloads/Avenue Dataset" \
        --baseline outputs/baseline.json --video 01
    python scripts/run_pipeline.py --data-root "~/Downloads/Avenue Dataset" \
        --baseline outputs/baseline.json --video all
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motion_agent_vad.baseline import MotionBaseline  # noqa: E402
from motion_agent_vad.config import Config  # noqa: E402
from motion_agent_vad.dataset import AvenueDataset  # noqa: E402
from motion_agent_vad.detection import Detector  # noqa: E402
from motion_agent_vad.pipeline import MotionAgentPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", default=str(root / "config" / "default.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--baseline", default=str(root / "outputs" / "baseline.json"))
    parser.add_argument(
        "--video",
        default="all",
        help="Test video name (e.g. '01') or 'all' to run every discovered test video",
    )
    parser.add_argument("--output-dir", default=str(root / "outputs"))
    parser.add_argument("--max-frames", type=int, default=None, help="Debug: cap frames processed per video")
    parser.add_argument("--no-video", action="store_true", help="Skip writing annotated .mp4 (faster)")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.data_root:
        cfg.set("dataset.root", args.data_root)

    ds = AvenueDataset.discover(cfg)
    if not ds.test_videos:
        raise SystemExit("No testing videos found. Run scripts/inspect_dataset.py to debug.")

    if args.video != "all":
        videos = [v for v in ds.test_videos if v.name == args.video]
        if not videos:
            available = ", ".join(v.name for v in ds.test_videos)
            raise SystemExit(f"Video '{args.video}' not found among test videos: {available}")
    else:
        videos = ds.test_videos

    baseline = MotionBaseline.load(args.baseline)

    out_dir = Path(args.output_dir)
    (out_dir / "evidence").mkdir(parents=True, exist_ok=True)
    (out_dir / "scores").mkdir(parents=True, exist_ok=True)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)

    detector = Detector(cfg)  # shared model instance across videos (faster: loads YOLO once)
    pipeline = MotionAgentPipeline(cfg, baseline=baseline, detector=detector)

    for v in videos:
        print(f"\n=== Processing test video: {v.name} ===")
        t0 = time.time()
        out_video = None if args.no_video else str(out_dir / "videos" / f"{v.name}_annotated.mp4")
        evidence = pipeline.run(v.path, output_video_path=out_video, max_frames=args.max_frames)
        dt = time.time() - t0

        evidence.to_csv(str(out_dir / "evidence" / f"{v.name}.csv"))
        evidence.to_json(str(out_dir / "evidence" / f"{v.name}.json"))

        regional_scores = getattr(evidence, "regional_only_scores", [])
        n_frames = getattr(evidence, "n_frames_processed", len(regional_scores))
        with open(out_dir / "scores" / f"{v.name}_regional_baseline.json", "w") as f:
            json.dump({"n_frames": n_frames, "scores": regional_scores}, f)

        print(
            f"  {n_frames} frames in {dt:.1f}s ({n_frames / max(dt, 1e-6):.1f} fps), "
            f"{len(evidence)} evidence records written"
        )
        if out_video:
            print(f"  annotated video -> {out_video}")
        print(f"  evidence -> {out_dir / 'evidence' / (v.name + '.csv')}")


if __name__ == "__main__":
    main()
