import math

from motion_agent_vad.deviation import compute_contextual_deviation, regional_only_score, z_score


def test_z_score_basic_formula():
    assert abs(z_score(x=10, mu=5, sigma=2, eps=1e-6) - 2.5) < 1e-5


def test_z_score_zero_sigma_uses_epsilon_no_division_by_zero():
    # would raise ZeroDivisionError if eps weren't applied; also must not
    # blow up to an uninformative astronomical value -- capped at max_z.
    val = z_score(x=3, mu=1, sigma=0, eps=1e-6, max_z=50.0)
    assert val == 50.0


def test_z_score_cap_is_configurable_and_only_applies_above_the_cap():
    # below the cap: unaffected
    assert abs(z_score(x=10, mu=5, sigma=2, eps=1e-6, max_z=100.0) - 2.5) < 1e-5
    # a genuinely huge deviation (near-zero training variance) gets clipped
    huge = z_score(x=100, mu=0, sigma=0, eps=1e-6, max_z=10.0)
    assert huge == 10.0


def test_fast_object_in_normally_fast_region_scores_lower_than_same_speed_in_calm_region():
    """The whole point of contextual deviation: raw speed alone should NOT
    determine the anomaly score -- it must be judged against what's normal
    for that specific region."""
    fast_region_dev = compute_contextual_deviation(
        object_speed=50.0,
        region_current_mag=48.0,
        region_mu=45.0,   # this region is normally busy/fast
        region_sigma=10.0,
        eps=1e-6,
        weight_object_motion=0.65,
        weight_regional_motion=0.35,
    )
    calm_region_dev = compute_contextual_deviation(
        object_speed=50.0,
        region_current_mag=48.0,
        region_mu=2.0,    # this region is normally still
        region_sigma=1.0,
        eps=1e-6,
        weight_object_motion=0.65,
        weight_regional_motion=0.35,
    )
    assert fast_region_dev.anomaly_score_raw < calm_region_dev.anomaly_score_raw


def test_object_matching_regional_norm_scores_near_zero():
    dev = compute_contextual_deviation(
        object_speed=10.0,
        region_current_mag=10.0,
        region_mu=10.0,
        region_sigma=2.0,
        eps=1e-6,
        weight_object_motion=0.65,
        weight_regional_motion=0.35,
    )
    assert dev.z_object < 1e-3
    assert dev.z_region < 1e-3
    assert dev.anomaly_score_raw < 1e-3


def test_regional_only_score_ignores_object_speed_entirely():
    # regional_only_score has no object_speed argument at all -- this test
    # just confirms it reduces to the region z-score used inside the
    # combined score, so the "baseline system" comparator is well-defined.
    s = regional_only_score(region_current_mag=8.0, region_mu=2.0, region_sigma=1.0, eps=1e-6)
    dev = compute_contextual_deviation(
        object_speed=999.0,  # irrelevant to this function
        region_current_mag=8.0,
        region_mu=2.0,
        region_sigma=1.0,
        eps=1e-6,
        weight_object_motion=0.65,
        weight_regional_motion=0.35,
    )
    assert abs(s - dev.z_region) < 1e-9
