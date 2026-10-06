import pytest

from motion_agent_vad.smoothing import EMASmoother


def test_ema_first_value_passes_through_unchanged():
    sm = EMASmoother(alpha=0.3)
    assert sm.update(1, 10.0) == 10.0


def test_ema_matches_manual_formula():
    sm = EMASmoother(alpha=0.4)
    s1 = sm.update(1, 10.0)
    s2 = sm.update(1, 20.0)
    expected_s2 = 0.4 * 20.0 + 0.6 * s1
    assert abs(s2 - expected_s2) < 1e-9


def test_ema_smooths_a_spike():
    sm = EMASmoother(alpha=0.2)
    for _ in range(10):
        sm.update(1, 1.0)
    spiked = sm.update(1, 100.0)
    # a single spike should be heavily damped, not pass through raw
    assert spiked < 100.0
    assert spiked > 1.0


def test_ema_is_independent_per_key():
    sm = EMASmoother(alpha=0.5)
    sm.update("a", 10.0)
    sm.update("a", 10.0)
    sm.update("b", 0.0)
    assert sm.value("a") == 10.0
    assert sm.value("b") == 0.0


def test_ema_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        EMASmoother(alpha=0.0)
    with pytest.raises(ValueError):
        EMASmoother(alpha=1.5)


def test_ema_reset():
    sm = EMASmoother(alpha=0.5)
    sm.update(1, 5.0)
    sm.reset(1)
    assert sm.value(1) is None
    # after reset, next update passes through unchanged again
    assert sm.update(1, 3.0) == 3.0
