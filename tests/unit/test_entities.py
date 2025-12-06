from domain.entities.track import Track
from domain.value_objects.verification_result import InferenceResult, MatchStatus


def test_track_duration_and_validity():
    track = Track(track_id=1, video_id="v1", frames=[1, 6, 16], bboxes=[(0, 0, 1, 1)], fps=30.0)
    assert track.duration_seconds == (16 - 1) / 30.0
    assert track.is_valid


def test_inference_result_helpers():
    res = InferenceResult(status=MatchStatus.CONFIRMED, confidence=0.9, reason="ok", raw_response="")
    assert res.is_match
    err = InferenceResult.error("fail")
    assert err.status == MatchStatus.AMBIGUOUS
    assert not err.is_match
