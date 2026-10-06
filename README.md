# Motion Agent for Video Anomaly Detection (VAD)

A motion-centric video anomaly detection pipeline for the CUHK Avenue
dataset. The core idea: **an object is not anomalous simply because it
moves fast — it is anomalous if its motion is unusual *for the region it is
currently in*.** A person sprinting across a road (normally busy/fast)
should not be flagged the same way as someone sprinting across a quiet
footpath (normally still).

```
Video Input
    |
Frame Preprocessing
    |
       Motion Agent
  +---------------+---------------+
  |                               |
YOLO Object Detection      Optical Flow (Farneback)
  |                               |
ByteTrack Tracking          8x8 Regional Motion
  |                               |
Object Motion Features            |
  +---------------+---------------+
                  |
      Contextual Motion Deviation
                  |
            Anomaly Score
                  |
        Temporal Smoothing (EMA)
                  |
          Evidence Output (CSV/JSON)
                  |
             Evaluation
```

---

## IMPORTANT — read this first

This project was built and its logic verified by an AI assistant (Claude)
that did **not** have access to your machine or to a copy of the CUHK
Avenue dataset. Concretely, here's exactly what was and wasn't done before
this was handed to you:

**Done, with real execution:**
- Every module was written as complete, working code — nothing here is a
  stub or a placeholder.
- The full unit test suite (`tests/`, 43 tests) runs against synthetic data
  and passes: optical-flow math, the 8x8 regional grid partitioning,
  Welford running mean/std for the baseline, the contextual z-score
  deviation formula, EMA smoothing, evidence CSV/JSON I/O, Avenue
  `.mat`/`.npy`/`.txt` ground-truth parsing, and precision/recall/F1/AUC
  (cross-checked against `scikit-learn`'s own implementations).
- The **entire CLI pipeline** (`inspect_dataset.py` -> `build_baseline.py`
  -> `run_pipeline.py` -> `evaluate.py`) was run end-to-end against a small
  *synthetic* stand-in dataset (simple moving shapes, not real Avenue
  footage, since the real dataset lives on your machine) with real YOLO +
  ByteTrack inference, and separately against real results from a user's
  first full run on the actual Avenue dataset. This caught and fixed three
  real bugs:
  1. The detector-independent "baseline system" score was silently always
     zero when no object was detected (see `pipeline.py`).
  2. Regions with ~0 training variance could produce astronomically large,
     meaningless z-scores — now capped via `deviation.max_z_score`.
  3. **A unit mismatch**: `object_speed` is computed in original-resolution
     pixels/second, but the regional baseline was being built from raw
     Farneback magnitude on the *resized* frame, *per frame-step* (not per
     second). Comparing the two directly inflated `z_object` almost every
     frame regardless of whether motion was actually unusual — this showed
     up as the "proposed" system's recall sitting at ~1.000 on nearly every
     test video with poor precision, on a real first run against the
     Avenue dataset. Fixed in
     `optical_flow.magnitude_to_original_pixels_per_second` (with a
     regression test); both `build_baseline.py` and `run_pipeline.py` now
     use matching units throughout.
- Because the synthetic smoke test used simple shapes, not people, YOLO
  produced zero real "person" detections during that particular run — so
  it alone did not exercise the full object-motion-feature code path with
  real detections. That path (`object_motion.py`, the
  `compute_contextual_deviation` call site in `pipeline.py`) is covered by
  direct unit tests with synthetic values, and by bug #3 above, which *was*
  found via a real run with real detections.

**If you built a baseline or ran the pipeline with an earlier copy of this
code**: the fix in point 3 changes the units the baseline is stored in.
**Rebuild `outputs/baseline.json`** (`scripts/build_baseline.py`) and
re-run `scripts/run_pipeline.py` / `scripts/evaluate.py` from scratch —
don't reuse an old baseline file or old evidence/report from before this
fix, the numbers won't be comparable.

**Not done — you need to do this:**
- **Running the pipeline on the actual CUHK Avenue dataset.** This needs
  your dataset files, which only exist on your Mac.
- **All accuracy/precision/recall/F1/AUC numbers reported anywhere in this
  README are placeholders you fill in** — see [Results](#results). No
  metric in this document is a real measurement; inventing one would
  violate the "actual measured results only" requirement of this project.

So: **run the four commands under [Quickstart](#quickstart) on your Mac**,
paste your own numbers into the Results section, and you have a fully
verified, fully your-own-results BTP deliverable.

---

## Architecture

### YOLO + ByteTrack (`detection.py`, `object_motion.py`)
[Ultralytics YOLO](https://docs.ultralytics.com/) (`yolov8n.pt` by default,
auto-downloaded on first run) detects people (COCO class 0). Ultralytics'
built-in ByteTrack (`model.track(..., tracker="bytetrack.yaml")`) assigns
persistent track IDs across frames. **YOLO itself never decides what's
anomalous** — it only supplies boxes + IDs. From each track's short
position history we derive:
- **Position** — bounding-box center per frame
- **Speed** — pixels/second between consecutive samples
- **Direction** — `atan2(dy, dx)`, in radians
- **Acceleration** — change in speed / dt, once >= 3 samples exist

### Optical Flow + 8x8 Regional Motion (`optical_flow.py`)
OpenCV's Farneback dense optical flow is computed between consecutive
grayscale frames, giving per-pixel `(dx, dy)`. This is converted to
magnitude/angle, and the frame is partitioned into an 8x8 grid (configurable
via `config/default.yaml: grid.rows/cols`); each cell's mean magnitude, std
magnitude, and magnitude-weighted circular-mean direction are computed.

### Normal Motion Baseline (`baseline.py`)
Built **only** from the training (normal) videos. For every one of the 64
regions, Welford's online algorithm accumulates a running mean and std of
that region's per-frame mean flow magnitude across every training video, in
one pass (no need to hold all frames in memory). Saved to `outputs/baseline.json`
and reloaded for every test run — build it once, evaluate as many times as
you like.

### Contextual Motion Deviation (`deviation.py`)
For each tracked person: find their bbox center -> map to an 8x8 region ->
look up that region's `(mu, sigma)` from the baseline -> compute two
z-scores:

```
z_object = |object_speed         - mu| / (sigma + eps)   # is THIS person's motion unusual here?
z_region = |current_regional_mag - mu| / (sigma + eps)   # is the whole region's flow unusual right now?

anomaly_score_raw = w_obj * z_object + w_reg * z_region
```

(weights configurable, default 0.65 / 0.35). Both z-scores are capped at
`deviation.max_z_score` (default 50) — otherwise a region with ~0 training
variance would produce meaningless, unbounded scores from a single small
deviation (this was caught during verification; see above).

### Temporal Smoothing (`smoothing.py`)
Each track's raw anomaly score is passed through an EMA:
`smoothed_t = alpha * raw_t + (1-alpha) * smoothed_{t-1}` (default
`alpha=0.3`), reducing frame-to-frame flicker. The detector-independent
regional-only score (used for the baseline comparator, see below) is
smoothed the same way, separately.

### Evidence Output (`evidence.py`)
Every tracked object, every frame, becomes one structured record: frame,
timestamp, track_id, bbox, region, speed, direction, acceleration, regional
motion (mean & current), both z-scores, raw and smoothed anomaly score,
detection confidence, and a boolean anomaly flag. Written to
`outputs/evidence/{video}.csv` and `.json`.

### Visualization (`visualization.py`)
The annotated output video overlays: YOLO boxes + ByteTrack IDs (colored by
anomaly score, red = high), a Farneback flow-arrow field, the 8x8 regional
motion heatmap, per-track trajectories, and a raw-vs-smoothed score bar.

### Evaluation (`evaluate.py`, `dataset.py`)
Loads Avenue's frame-level ground truth (from the `.mat` pixel-mask
volumes in `ground_truth_demo/testing_label_mask/`, or `.npy`/`.txt` if you
provide your own) and compares, on the exact same test videos:
- **Baseline system**: optical flow -> regional motion -> anomaly score
  (no detection/tracking at all — the whole-grid regional z-score, computed
  every frame regardless of whether anyone was detected)
- **Proposed system**: YOLO -> ByteTrack -> object motion, combined with
  regional motion -> contextual motion deviation -> anomaly score

Reports precision/recall/F1 at a fixed threshold, best achievable F1 (via
threshold sweep), and ROC-AUC, per video and pooled.

---

## Installation

Requires Python 3.10+ (tested with 3.11/3.12). On a MacBook Air M4:

```bash
cd motion_agent_vad
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`ultralytics` will auto-download `yolov8n.pt` (~6 MB) the first time you run
detection. PyTorch will use Apple's MPS backend automatically on your M4
(`detection.device: "auto"` in the config resolves to `"mps"` — see
`config.py: resolve_device`); no extra setup needed.

## Dataset setup

1. Make sure the CUHK Avenue Dataset is unzipped somewhere, e.g.
   `~/Downloads/Avenue Dataset/`.
2. Point the pipeline at it, either by editing `config/default.yaml`:
   ```yaml
   dataset:
     root: "~/Downloads/Avenue Dataset"
   ```
   or by passing `--data-root` to any script (overrides the config).
3. Check what was actually discovered:
   ```bash
   python scripts/inspect_dataset.py --data-root "~/Downloads/Avenue Dataset"
   ```
   This prints the training/testing video folders it found, how many
   videos and frames, and whether ground truth was located. Avenue is
   distributed under a couple of different folder-naming conventions
   depending on source, so the loader searches several candidate names
   (`dataset.train_dir_candidates` / `test_dir_candidates` /
   `gt_dir_candidates` in the config) — if it can't find something, add
   your actual folder name to the relevant candidate list.

## Quickstart

```bash
# 1. Inspect the dataset — confirm training/testing videos & ground truth are found
python scripts/inspect_dataset.py --data-root "~/Downloads/Avenue Dataset"

# 2. Build the normal-motion baseline from the TRAINING videos only
python scripts/build_baseline.py --data-root "~/Downloads/Avenue Dataset" \
    --output outputs/baseline.json

# 3. Run the full pipeline on all test videos (drop --video all -> a name like "01" for just one)
python scripts/run_pipeline.py --data-root "~/Downloads/Avenue Dataset" \
    --baseline outputs/baseline.json --video all

# 4. Evaluate proposed vs. baseline against ground truth
python scripts/evaluate.py --data-root "~/Downloads/Avenue Dataset" --output-dir outputs
```

Step 3 is the slow one (YOLO inference per frame). Useful flags:
- `--video 01` — run just one test video first, to sanity-check before
  committing to the full test set.
- `--max-frames 200` — cap frames per video, for a quick end-to-end dry run.
- `--no-video` — skip writing the annotated `.mp4` (evidence CSV/JSON still
  written); meaningfully faster.

Outputs land under `outputs/`:
```
outputs/
├── baseline.json                       per-region normal-motion stats
├── evidence/{video}.csv, .json         per-object structured evidence
├── scores/{video}_regional_baseline.json   per-frame baseline-system score
├── videos/{video}_annotated.mp4        annotated visualization
└── evaluation_report.md, .json         precision/recall/F1/AUC, both systems
```

## Running tests

```bash
pip install pytest
pytest tests/ -v
```
These are pure-logic tests against synthetic data (no dataset or YOLO
weights required) — they check the math (Welford stats, z-scores, EMA,
grid partitioning, metric formulas against `scikit-learn`) and the
ground-truth parsers, independent of what's in your copy of Avenue.

## Configuration

Everything tunable lives in `config/default.yaml` (grid size, Farneback
parameters, YOLO confidence/classes, EMA alpha, deviation weights, the
anomaly threshold, output options, dataset folder candidates...). Copy it
and pass `--config your_copy.yaml` to any script rather than editing
defaults in place, if you want to keep multiple configurations around.

## Results

*(Fill this in after running Quickstart step 4 on the real dataset —
paste the contents of `outputs/evaluation_report.md` here, or attach it
alongside this README. Do not report numbers that weren't actually
measured by the scripts in this repo.)*

```
<paste outputs/evaluation_report.md here>
```

A few things worth commenting on once you have real numbers:
- Does the proposed (contextual) system beat the region-only baseline on
  F1 / AUC? By how much?
- Look at a handful of `is_anomalous=True` rows in `evidence/*.csv` for a
  test video with known ground-truth anomalies — do the flagged regions
  and objects visually make sense in the annotated video?
- Where does the proposed system do worse than the baseline, if anywhere,
  and why (e.g., YOLO missing small/occluded people; ByteTrack ID
  switches breaking the motion history)?

## Limitations

- **YOLO is a person detector**: anomalies that aren't people (e.g. a
  thrown object, a bicycle where none belong) are invisible to the
  proposed system's *object* motion features, though they can still show
  up via the regional-motion component.
- **CPU/MPS-only, no batching across videos**: this is a from-scratch
  research-style pipeline, not optimized for throughput; on an M4 expect
  roughly real-time-to-a-few-times-slower-than-real-time processing
  depending on resolution (tune `optical_flow.resize_longer_side` and
  `output.frame_stride` if you need it faster).
- **ByteTrack ID switches**: occlusion or fast motion can cause a person to
  get a new track ID mid-sequence, which resets their motion history
  (a fresh track has no speed/acceleration until a few frames accumulate).
- **Region granularity is fixed at frame-partition level (8x8)**: a person
  near a region boundary can flicker between two regions' baselines frame
  to frame; a finer or overlapping grid would smooth this at the cost of
  more compute and a sparser per-region baseline.
- **The baseline assumes stationarity**: it's a single mean/std per region
  over the whole training set, not time-of-day/lighting-conditional; Avenue
  is reasonably static across its own train/test split, but this would not
  transfer as-is to a scene with strong diurnal variation.
- **z-score capping (`max_z_score`)** is a pragmatic fix for near-zero
  training variance, not a first-principles solution — a small sigma-floor
  or a Bayesian regularized variance estimate would be a more principled
  alternative if this matters for your report's methodology section.

## Project layout

```
motion_agent_vad/
├── README.md
├── requirements.txt
├── config/
│   └── default.yaml
├── src/motion_agent_vad/
│   ├── config.py          # YAML config loader, device resolution
│   ├── dataset.py          # Avenue discovery + ground-truth parsing
│   ├── optical_flow.py     # Farneback flow + 8x8 regional grid
│   ├── baseline.py         # Welford running stats, save/load
│   ├── detection.py        # YOLO + ByteTrack wrapper
│   ├── object_motion.py    # per-track speed/direction/acceleration
│   ├── deviation.py        # contextual z-score anomaly scoring
│   ├── smoothing.py        # EMA
│   ├── evidence.py         # structured records, CSV/JSON export
│   ├── visualization.py    # annotated-video drawing helpers
│   ├── pipeline.py         # orchestrates everything, per video
│   └── evaluate.py         # precision/recall/F1/AUC, system comparison
├── scripts/
│   ├── inspect_dataset.py
│   ├── build_baseline.py
│   ├── run_pipeline.py
│   └── evaluate.py
├── tests/                  # 40 unit tests, synthetic data only
├── outputs/                 # generated (baseline, evidence, videos, report)
└── data/                     # (empty; point config at your own dataset location)
```
