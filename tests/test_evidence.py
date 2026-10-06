import csv
import json

from motion_agent_vad.evidence import EvidenceRecord, EvidenceWriter


def _sample_record(frame=0, track_id=1, score=1.0):
    return EvidenceRecord(
        frame=frame,
        timestamp=frame / 25.0,
        track_id=track_id,
        bbox_x1=1.0,
        bbox_y1=2.0,
        bbox_x2=3.0,
        bbox_y2=4.0,
        region_row=0,
        region_col=0,
        speed=5.0,
        direction=0.5,
        acceleration=0.1,
        regional_motion_mean=1.0,
        regional_motion_current=1.2,
        z_object=0.5,
        z_region=0.3,
        contextual_deviation=score,
        anomaly_score_raw=score,
        anomaly_score_smoothed=score,
        confidence=0.9,
        is_anomalous=score >= 2.5,
    )


def test_writer_csv_roundtrip(tmp_path):
    w = EvidenceWriter()
    w.add(_sample_record(frame=0, track_id=1, score=1.0))
    w.add(_sample_record(frame=1, track_id=1, score=3.0))
    path = tmp_path / "ev.csv"
    w.to_csv(str(path))

    with open(path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[1]["is_anomalous"] == "True"
    assert float(rows[1]["anomaly_score_smoothed"]) == 3.0


def test_writer_json_roundtrip(tmp_path):
    w = EvidenceWriter()
    w.add(_sample_record(frame=0, track_id=7, score=2.0))
    path = tmp_path / "ev.json"
    w.to_json(str(path))

    with open(path) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["track_id"] == 7
    assert data[0]["anomaly_score_smoothed"] == 2.0


def test_empty_writer_still_writes_header_only_csv(tmp_path):
    w = EvidenceWriter()
    path = tmp_path / "empty.csv"
    w.to_csv(str(path))
    with open(path) as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1  # header only
    assert "track_id" in rows[0]


def test_frame_level_scores_takes_max_per_frame_and_zero_fills_gaps():
    w = EvidenceWriter()
    w.add(_sample_record(frame=0, track_id=1, score=1.0))
    w.add(_sample_record(frame=0, track_id=2, score=4.0))  # two tracks in same frame
    w.add(_sample_record(frame=2, track_id=1, score=0.5))
    scores = w.frame_level_scores(n_frames=4)
    assert scores == [4.0, 0.0, 0.5, 0.0]
