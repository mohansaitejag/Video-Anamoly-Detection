import math

from motion_agent_vad.object_motion import TrackHistory, bbox_center


def test_bbox_center():
    assert bbox_center((0, 0, 10, 20)) == (5.0, 10.0)


def test_no_history_returns_zero_motion():
    th = TrackHistory(history_length=5)
    feats = th.features_for(track_id=1)
    assert feats.speed == 0.0
    assert feats.acceleration is None


def test_speed_and_direction_from_two_points():
    th = TrackHistory(history_length=5)
    th.update(track_id=1, frame_idx=0, t=0.0, cx=0.0, cy=0.0)
    th.update(track_id=1, frame_idx=1, t=1.0, cx=3.0, cy=4.0)  # moved (3,4) in 1s
    feats = th.features_for(track_id=1)
    assert abs(feats.speed - 5.0) < 1e-6  # 3-4-5 triangle, dt=1
    expected_dir = math.atan2(4.0, 3.0) % (2 * math.pi)
    assert abs(feats.direction - expected_dir) < 1e-6
    assert feats.acceleration is None  # only 2 samples so far


def test_acceleration_from_three_points():
    th = TrackHistory(history_length=5)
    th.update(track_id=1, frame_idx=0, t=0.0, cx=0.0, cy=0.0)
    th.update(track_id=1, frame_idx=1, t=1.0, cx=1.0, cy=0.0)   # speed 1 px/s
    th.update(track_id=1, frame_idx=2, t=2.0, cx=4.0, cy=0.0)   # speed 3 px/s
    feats = th.features_for(track_id=1)
    assert abs(feats.speed - 3.0) < 1e-6
    assert feats.acceleration is not None
    assert abs(feats.acceleration - 2.0) < 1e-6  # (3-1)/1


def test_prune_removes_inactive_tracks():
    th = TrackHistory(history_length=5)
    th.update(track_id=1, frame_idx=0, t=0.0, cx=0.0, cy=0.0)
    th.update(track_id=2, frame_idx=0, t=0.0, cx=1.0, cy=1.0)
    th.prune(active_ids=[1])
    assert len(th.get(1)) == 1
    assert len(th.get(2)) == 0


def test_history_length_is_bounded():
    th = TrackHistory(history_length=3)
    for i in range(10):
        th.update(track_id=1, frame_idx=i, t=float(i), cx=float(i), cy=0.0)
    assert len(th.get(1)) == 3
