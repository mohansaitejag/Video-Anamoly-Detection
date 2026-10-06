#!/usr/bin/env python3
"""Inspect the CUHK Avenue dataset: what was found, where, and basic stats.

Usage:
    python scripts/inspect_dataset.py --data-root "~/Downloads/Avenue Dataset"
    python scripts/inspect_dataset.py --config config/default.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motion_agent_vad.config import Config  # noqa: E402
from motion_agent_vad.dataset import AvenueDataset, get_video_frame_count  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config" / "default.yaml"))
    parser.add_argument("--data-root", default=None, help="Override dataset.root from the config")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.data_root:
        cfg.set("dataset.root", args.data_root)

    ds = AvenueDataset.discover(cfg)
    print(ds.summary())
    print()

    if ds.train_videos:
        print("Training videos:")
        for v in ds.train_videos[:5]:
            n = get_video_frame_count(v.path)
            print(f"  {v.name:>6s}  {n:>6d} frames  {v.path}")
        if len(ds.train_videos) > 5:
            print(f"  ... and {len(ds.train_videos) - 5} more")
    else:
        print("No training videos found. Check dataset.root / train_dir_candidates in your config.")

    print()
    if ds.test_videos:
        print("Testing videos:")
        for v in ds.test_videos[:5]:
            n = get_video_frame_count(v.path)
            print(f"  {v.name:>6s}  {n:>6d} frames  {v.path}")
        if len(ds.test_videos) > 5:
            print(f"  ... and {len(ds.test_videos) - 5} more")
    else:
        print("No testing videos found. Check dataset.root / test_dir_candidates in your config.")

    print()
    if not ds.gt_dir:
        print(
            "No ground-truth directory found. Evaluation will be skipped for any "
            "video without matching ground truth (see gt_dir_candidates in your config)."
        )
    else:
        print(f"Ground truth directory: {ds.gt_dir}")


if __name__ == "__main__":
    main()
