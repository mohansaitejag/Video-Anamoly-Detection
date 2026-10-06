import numpy as np

from motion_agent_vad.evaluate import compare_systems, evaluate_scores


def test_evaluate_scores_matches_sklearn_reference():
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    rng = np.random.default_rng(0)
    labels = (rng.random(200) > 0.7).astype(int)
    scores = labels * rng.normal(4, 1, 200) + (1 - labels) * rng.normal(1, 1, 200)
    threshold = 2.5

    m = evaluate_scores(scores.tolist(), labels.tolist(), fixed_threshold=threshold)
    preds = (scores >= threshold).astype(int)

    assert abs(m.precision - precision_score(labels, preds, zero_division=0)) < 1e-9
    assert abs(m.recall - recall_score(labels, preds, zero_division=0)) < 1e-9
    assert abs(m.f1 - f1_score(labels, preds, zero_division=0)) < 1e-9
    assert m.auc is not None
    assert abs(m.auc - roc_auc_score(labels, scores)) < 1e-9


def test_evaluate_scores_perfect_separation_gives_f1_one_at_best_threshold():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.1, 0.2, 0.3, 5.0, 6.0, 7.0]
    m = evaluate_scores(scores, labels, fixed_threshold=3.0)
    assert m.f1 == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.best_f1 == 1.0
    assert m.auc == 1.0


def test_evaluate_scores_no_positive_frames_auc_is_none():
    labels = [0, 0, 0, 0]
    scores = [0.1, 0.5, 0.2, 0.9]
    m = evaluate_scores(scores, labels, fixed_threshold=1.0)
    assert m.auc is None
    assert m.n_positive == 0


def test_compare_systems_pools_correctly_and_skips_videos_without_ground_truth():
    per_video_scores = {
        "v1": {"baseline": [0.1, 0.2, 5.0, 5.0], "proposed": [0.1, 0.1, 6.0, 6.0]},
        "v2": {"baseline": [0.1, 5.0], "proposed": [0.1, 6.0]},
        "v3_no_gt": {"baseline": [1.0, 2.0], "proposed": [1.0, 2.0]},
    }
    per_video_labels = {
        "v1": [0, 0, 1, 1],
        "v2": [0, 1],
        # v3_no_gt intentionally omitted
    }
    report = compare_systems(per_video_scores, per_video_labels, fixed_threshold=2.5)

    assert report.per_video["v1"]["proposed"]["f1"] == 1.0
    assert report.per_video["v3_no_gt"]["baseline"] is None
    assert report.per_video["v3_no_gt"]["proposed"] is None

    # pooled should combine v1 + v2 only (6 frames total), not v3
    assert report.pooled["proposed"]["n_frames"] == 6
    md = report.to_markdown()
    assert "proposed" in md and "baseline" in md
