#!/usr/bin/env python3
"""Evaluate the proposed (contextual) system against the baseline
(optical-flow-only) system, using CUHK Avenue ground truth, for whichever
test videos scripts/run_pipeline.py has already produced output for.

Usage:
    python scripts/evaluate.py --data-root "~/Downloads/Avenue Dataset" \
        --output-dir outputs
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from motion_agent_vad.config import Config  # noqa: E402
from motion_agent_vad.dataset import AvenueDataset, load_frame_level_ground_truth  # noqa: E402
from motion_agent_vad.evaluate import compare_systems  # noqa: E402


def _proposed_scores_from_evidence(evidence_json_path: Path, n_frames: int):
    if not evidence_json_path.exists():
        return None
    with open(evidence_json_path) as f:
        records = json.load(f)
    scores = [0.0] * n_frames
    for r in records:
        fr = r["frame"]
        if 0 <= fr < n_frames:
            scores[fr] = max(scores[fr], r["anomaly_score_smoothed"])
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", default=str(root / "config" / "default.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=str(root / "outputs"))
    parser.add_argument("--report", default=None, help="Output report path prefix (default: {output-dir}/evaluation_report)")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    if args.data_root:
        cfg.set("dataset.root", args.data_root)

    ds = AvenueDataset.discover(cfg)
    out_dir = Path(args.output_dir)

    per_video_scores = {}
    per_video_labels = {}
    skipped = []

    for v in ds.test_videos:
        score_meta_path = out_dir / "scores" / f"{v.name}_regional_baseline.json"
        evidence_json_path = out_dir / "evidence" / f"{v.name}.json"
        if not score_meta_path.exists() or not evidence_json_path.exists():
            continue  # this video hasn't been run through scripts/run_pipeline.py

        with open(score_meta_path) as f:
            meta = json.load(f)
        n_frames = meta["n_frames"]
        baseline_scores = meta["scores"]
        proposed_scores = _proposed_scores_from_evidence(evidence_json_path, n_frames)

        labels = load_frame_level_ground_truth(v, ds.gt_dir, n_frames)
        if labels is None:
            skipped.append(v.name)
            continue

        per_video_scores[v.name] = {"baseline": baseline_scores, "proposed": proposed_scores}
        per_video_labels[v.name] = labels.tolist()

    if not per_video_scores:
        raise SystemExit(
            "No videos with both pipeline output AND matching ground truth were found.\n"
            "Run scripts/run_pipeline.py first, and check that ground truth is discoverable "
            "(see scripts/inspect_dataset.py)."
        )

    report = compare_systems(per_video_scores, per_video_labels, fixed_threshold=cfg.anomaly.score_threshold)

    report_prefix = args.report or str(out_dir / "evaluation_report")
    report.to_json(report_prefix + ".json")
    md = report.to_markdown()
    with open(report_prefix + ".md", "w") as f:
        f.write(md)

    print(md)
    if skipped:
        print(f"\n(Skipped {len(skipped)} video(s) with no matching ground truth: {', '.join(skipped)})")
    print(f"\nSaved: {report_prefix}.json and {report_prefix}.md")


if __name__ == "__main__":
    main()
