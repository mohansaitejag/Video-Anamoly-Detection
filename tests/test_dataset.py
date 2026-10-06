import numpy as np
import pytest

scipy_io = pytest.importorskip("scipy.io")

from motion_agent_vad.dataset import VideoEntry, load_frame_level_ground_truth


def test_ground_truth_from_mat_object_array_of_masks(tmp_path):
    """Mirrors the real Avenue format: a MATLAB cell array (object dtype in
    scipy) of per-frame pixel masks, one mask per frame."""
    n_frames = 6
    masks = np.empty((n_frames,), dtype=object)
    for i in range(n_frames):
        mask = np.zeros((20, 20), dtype=np.uint8)
        if i in (2, 3):  # frames 2-3 are anomalous
            mask[5:10, 5:10] = 1
        masks[i] = mask

    scipy_io.savemat(tmp_path / "01_label.mat", {"volLabel": masks})

    video = VideoEntry(path="unused", name="01")
    labels = load_frame_level_ground_truth(video, str(tmp_path), n_frames)
    assert labels is not None
    assert labels.tolist() == [0, 0, 1, 1, 0, 0]


def test_ground_truth_from_mat_3d_volume(tmp_path):
    """Some distributions store ground truth as a (H, W, T) numeric volume
    instead of a MATLAB cell array."""
    h, w, t = 10, 10, 5
    vol = np.zeros((h, w, t), dtype=np.uint8)
    vol[:, :, 4] = 1  # last frame anomalous
    scipy_io.savemat(tmp_path / "02_label.mat", {"volLabel": vol})

    video = VideoEntry(path="unused", name="02")
    labels = load_frame_level_ground_truth(video, str(tmp_path), n_frames=t)
    assert labels.tolist() == [0, 0, 0, 0, 1]


def test_ground_truth_from_npy(tmp_path):
    arr = np.array([0, 1, 1, 0])
    np.save(tmp_path / "03.npy", arr)
    video = VideoEntry(path="unused", name="03")
    labels = load_frame_level_ground_truth(video, str(tmp_path), n_frames=4)
    assert labels.tolist() == [0, 1, 1, 0]


def test_ground_truth_from_txt_ranges(tmp_path):
    (tmp_path / "04.txt").write_text("3-5\n8-8\n")
    video = VideoEntry(path="unused", name="04")
    labels = load_frame_level_ground_truth(video, str(tmp_path), n_frames=10)
    # 1-indexed inclusive ranges -> frames 2,3,4 and 7 (0-indexed)
    expected = [0, 0, 1, 1, 1, 0, 0, 1, 0, 0]
    assert labels.tolist() == expected


def test_ground_truth_missing_returns_none(tmp_path):
    video = VideoEntry(path="unused", name="nope")
    assert load_frame_level_ground_truth(video, str(tmp_path), n_frames=10) is None
    assert load_frame_level_ground_truth(video, None, n_frames=10) is None
