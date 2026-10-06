import numpy as np

from motion_agent_vad.optical_flow import (
    FlowParams,
    RegionalMotionGrid,
    compute_farneback_flow,
    flow_to_magnitude_angle,
    magnitude_to_original_pixels_per_second,
    maybe_resize,
    to_grayscale,
)


def _shifted_pair(shift=6, size=128):
    """A textured image and a copy shifted right by `shift` px, so Farneback
    should recover a clear rightward flow."""
    rng = np.random.default_rng(0)
    base = (rng.random((size + shift, size + shift)) * 255).astype(np.uint8)
    # add some blocky structure so Farneback has texture to track
    base[::8, :] = 255
    base[:, ::8] = 0
    prev = base[:size, :size]
    curr = base[:size, shift : shift + size]
    return prev, curr


def test_flow_detects_rightward_shift():
    prev, curr = _shifted_pair(shift=6)
    flow = compute_farneback_flow(prev, curr, FlowParams())
    assert flow.shape == (128, 128, 2)
    mean_dx = flow[..., 0].mean()
    mean_dy = flow[..., 1].mean()
    # curr is `prev shifted right by 6px`, i.e. content moved left relative
    # to a fixed window -> Farneback should report a consistent horizontal
    # flow of non-trivial magnitude in one direction.
    assert abs(mean_dx) > 1.0
    assert abs(mean_dy) < abs(mean_dx)


def test_magnitude_angle_shapes_and_ranges():
    flow = np.zeros((32, 32, 2), dtype=np.float32)
    flow[..., 0] = 3.0
    flow[..., 1] = 4.0
    mag, ang = flow_to_magnitude_angle(flow)
    assert mag.shape == (32, 32)
    assert ang.shape == (32, 32)
    assert np.allclose(mag, 5.0, atol=1e-4)  # 3-4-5 triangle
    assert np.all(ang >= 0) and np.all(ang <= 2 * np.pi)


def test_regional_grid_partitions_frame_without_gaps_or_overlap():
    grid = RegionalMotionGrid(rows=8, cols=8)
    h, w = 64, 64
    mag = np.ones((h, w), dtype=np.float32)
    ang = np.zeros((h, w), dtype=np.float32)
    mean_mag, std_mag, mean_ang = grid.compute_regional_stats(mag, ang)
    assert mean_mag.shape == (8, 8)
    # uniform input -> every cell should see mean magnitude == 1, std == 0
    assert np.allclose(mean_mag, 1.0)
    assert np.allclose(std_mag, 0.0)


def test_region_index_for_point_is_within_grid_bounds():
    grid = RegionalMotionGrid(rows=8, cols=8)
    for x, y in [(0, 0), (639, 359), (320, 180), (1e6, 1e6), (-5, -5)]:
        row, col = grid.region_index_for_point(x, y, frame_w=640, frame_h=360)
        assert 0 <= row < 8
        assert 0 <= col < 8


def test_maybe_resize_keeps_aspect_ratio_and_shrinks_only():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    resized, scale = maybe_resize(frame, longer_side=320)
    assert scale < 1.0
    assert max(resized.shape[:2]) == 320
    orig_ratio = 640 / 360
    new_ratio = resized.shape[1] / resized.shape[0]
    assert abs(orig_ratio - new_ratio) < 0.05

    # frame already smaller than target -> no resize
    small = np.zeros((100, 100, 3), dtype=np.uint8)
    resized2, scale2 = maybe_resize(small, longer_side=320)
    assert scale2 == 1.0
    assert resized2.shape == small.shape


def test_magnitude_unit_conversion_matches_object_speed_units():
    """Regression test for a real bug: regional flow magnitude (computed on
    a resized frame, per Farneback frame-step) must be converted into the
    SAME units as object_speed (original-resolution pixels/second) before
    either is compared against the other or against the baseline. Otherwise
    z_object is comparing incommensurable quantities and saturates almost
    unconditionally (observed as near-100% recall / poor precision when this
    conversion was missing).
    """
    # A flow of 2 px/frame-step measured on a frame resized to half scale,
    # at 25 fps, corresponds to 4 px/frame-step at original resolution,
    # i.e. 100 px/second at original resolution.
    mag = np.full((4, 4), 2.0, dtype=np.float64)
    converted = magnitude_to_original_pixels_per_second(mag, scale=0.5, fps=25.0)
    assert np.allclose(converted, 100.0)


def test_magnitude_unit_conversion_no_resize_no_op_on_scale_only():
    # scale=1.0 (no resize) still converts per-frame -> per-second via fps
    mag = np.full((2, 2), 3.0)
    converted = magnitude_to_original_pixels_per_second(mag, scale=1.0, fps=10.0)
    assert np.allclose(converted, 30.0)


def test_magnitude_unit_conversion_guards_against_zero_scale():
    mag = np.array([[1.0]])
    # must not raise ZeroDivisionError / produce inf
    converted = magnitude_to_original_pixels_per_second(mag, scale=0.0, fps=25.0)
    assert np.isfinite(converted).all()


def test_to_grayscale_handles_color_and_gray_input():
    color = np.zeros((10, 10, 3), dtype=np.uint8)
    gray = to_grayscale(color)
    assert gray.shape == (10, 10)

    already_gray = np.zeros((10, 10), dtype=np.uint8)
    assert to_grayscale(already_gray).shape == (10, 10)
