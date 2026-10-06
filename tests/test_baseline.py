import numpy as np

from motion_agent_vad.baseline import BaselineBuilder, MotionBaseline


def test_baseline_builder_matches_numpy_mean_std():
    rng = np.random.default_rng(42)
    rows, cols, n_frames = 8, 8, 200
    frames = rng.normal(loc=2.0, scale=0.5, size=(n_frames, rows, cols))

    builder = BaselineBuilder(rows, cols)
    for i in range(n_frames):
        builder.update(frames[i])
    baseline = builder.finalize()

    expected_mean = frames.mean(axis=0)
    expected_std = frames.std(axis=0, ddof=1)

    assert np.allclose(baseline.mean, expected_mean, atol=1e-8)
    assert np.allclose(baseline.std, expected_std, atol=1e-6)
    assert np.all(baseline.n_samples == n_frames)


def test_baseline_single_sample_has_zero_std_not_nan():
    builder = BaselineBuilder(2, 2)
    builder.update(np.array([[1.0, 2.0], [3.0, 4.0]]))
    baseline = builder.finalize()
    assert np.all(baseline.std == 0.0)
    assert not np.any(np.isnan(baseline.std))


def test_baseline_save_and_load_roundtrip(tmp_path):
    builder = BaselineBuilder(4, 4)
    rng = np.random.default_rng(1)
    for _ in range(10):
        builder.update(rng.random((4, 4)))
    baseline = builder.finalize()

    path = tmp_path / "baseline.json"
    baseline.save(str(path))
    loaded = MotionBaseline.load(str(path))

    assert loaded.rows == baseline.rows
    assert loaded.cols == baseline.cols
    assert np.allclose(loaded.mean, baseline.mean)
    assert np.allclose(loaded.std, baseline.std)

    mu, sigma = loaded.stats_for_region(1, 2)
    assert mu == float(baseline.mean[1, 2])
    assert sigma == float(baseline.std[1, 2])
