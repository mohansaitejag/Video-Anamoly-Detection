#!/usr/bin/env python3
"""Build the per-region normal-motion baseline from the CUHK Avenue TRAINING
(normal) videos only, and save it for reuse during testing.

Usage:
    python scripts/build_baseline.py --data-root "~/Downloads/Avenue Dataset" \
        --output outputs/baseline.json
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motion_agent_vad.config import Config  # noqa: E402
from motion_agent_vad.dataset import AvenueDataset  # noqa: E402
from motion_agent_vad.pipeline import build_baseline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", default=str(root / "config" / "default.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output", default=str(root / "outputs" / "baseline.json"))
    parser.add_argument(
        "--limit", type=int, default=None, help="Only use the first N training videos (useful for a quick test)"
    )
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.data_root:
        cfg.set("dataset.root", args.data_root)

    ds = AvenueDataset.discover(cfg)
    if not ds.train_videos:
        raise SystemExit(
            "No training videos found. Run scripts/inspect_dataset.py to debug "
            "dataset discovery, or set --data-root / dataset.root correctly."
        )

    videos = ds.train_videos if args.limit is None else ds.train_videos[: args.limit]
    print(f"Building baseline from {len(videos)} training video(s)...")

    def progress(video_path, n_frames):
        print(f"  processed {Path(video_path).name}: {n_frames} frames")

    baseline = build_baseline([v.path for v in videos], cfg, progress_cb=progress)
    baseline.save(args.output)
    print(f"\nSaved baseline to {args.output}")
    print(f"Grid: {baseline.rows}x{baseline.cols}")
    print(f"Mean regional magnitude range: [{baseline.mean.min():.4f}, {baseline.mean.max():.4f}]")
    print(f"Std  regional magnitude range: [{baseline.std.min():.4f}, {baseline.std.max():.4f}]")


if __name__ == "__main__":
    main()
