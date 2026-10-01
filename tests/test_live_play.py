from __future__ import annotations

import pytest

from football_highlights.live_play import (
    FrameSignal,
    LivePlayConfig,
    ScorebugReading,
    sample_video_around_candidate,
    validate_live_play,
)


def accepted_frames() -> list[FrameSignal]:
    return [
        FrameSignal(0.0, green_field_ratio=0.78, motion_score=0.08, scorebug_visible=True),
        FrameSignal(0.5, green_field_ratio=0.76, motion_score=0.10, scorebug_visible=True),
        FrameSignal(1.0, green_field_ratio=0.75, motion_score=0.12, scorebug_visible=True),
        FrameSignal(
            1.5,
            green_field_ratio=0.73,
            motion_score=0.78,
            coordinated_motion_score=0.82,
            scorebug_visible=True,
        ),
        FrameSignal(
            2.0,
            green_field_ratio=0.70,
            motion_score=0.65,
            coordinated_motion_score=0.70,
            scorebug_visible=True,
        ),
        FrameSignal(
            2.5,
            green_field_ratio=0.68,
            motion_score=0.52,
            coordinated_motion_score=0.55,
            scorebug_visible=True,
        ),
        FrameSignal(
            3.0,
            green_field_ratio=0.65,
            motion_score=0.35,
            coordinated_motion_score=0.40,
            scorebug_visible=True,
        ),
        FrameSignal(3.5, green_field_ratio=0.63, motion_score=0.08, scorebug_visible=True),
        FrameSignal(4.0, green_field_ratio=0.61, motion_score=0.06, scorebug_visible=True),
    ]


def accepted_clock() -> list[ScorebugReading]:
    return [
        ScorebugReading(0.0, clock_seconds=600, quarter=1),
        ScorebugReading(1.5, clock_seconds=599, quarter=1),
        ScorebugReading(3.5, clock_seconds=598, quarter=1),
    ]


def test_accepts_a_real_live_play_when_all_hard_gates_pass() -> None:
    result = validate_live_play(accepted_frames(), accepted_clock(), candidate_timestamp=1.8)

    assert result.accepted
    assert result.confidence > 0.8
    assert result.live_start == pytest.approx(0.0)
    assert result.live_end == pytest.approx(4.0)
    assert result.snap_timestamp == pytest.approx(1.5)
    assert "accepted: all strict live-play gates passed" in result.reasons
    assert any("game clock moved" in reason for reason in result.reasons)


def test_rejects_commentary_without_scorebug_and_formation() -> None:
    frames = [
        FrameSignal(0.0, green_field_ratio=0.15, motion_score=0.10, scorebug_visible=False),
        FrameSignal(
            0.5,
            green_field_ratio=0.12,
            motion_score=0.70,
            coordinated_motion_score=0.80,
            scorebug_visible=False,
        ),
        FrameSignal(1.0, green_field_ratio=0.10, motion_score=0.40, scorebug_visible=False),
    ]
    result = validate_live_play(
        frames,
        [ScorebugReading(0.5, visible=False, clock_seconds=599, quarter=1)],
    )

    assert not result.accepted
    assert result.live_start is None
    assert result.live_end is None
    assert any("no stable wide green-field" in reason for reason in result.reasons)
    assert any("no coordinated motion/snap" in reason for reason in result.reasons)


def test_rejects_replay_cut_after_real_looking_pre_snap() -> None:
    frames = accepted_frames()
    frames[6] = FrameSignal(
        3.0,
        green_field_ratio=0.20,
        motion_score=0.40,
        shot_change_score=0.95,
        scorebug_visible=False,
        replay_indicator=True,
    )
    result = validate_live_play(frames, accepted_clock(), candidate_timestamp=1.8)

    assert not result.accepted
    assert result.live_end == pytest.approx(3.0)
    assert any("replay/shot transition" in reason for reason in result.reasons)
    assert any(
        "continuous live shot" in reason or "replay/shot" in reason for reason in result.reasons
    )


def test_rejects_clock_that_is_static_even_if_video_looks_like_live_action() -> None:
    static_clock = [
        ScorebugReading(0.0, clock_seconds=600, quarter=1),
        ScorebugReading(1.5, clock_seconds=600, quarter=1),
        ScorebugReading(3.5, clock_seconds=600, quarter=1),
    ]
    result = validate_live_play(accepted_frames(), static_clock, candidate_timestamp=1.8)

    assert not result.accepted
    assert any("game clock did not move" in reason for reason in result.reasons)
    assert dict(result.evidence)["game_clock_movement"] == 0.0


def test_rejects_uncoordinated_motion_after_formation() -> None:
    frames = accepted_frames()
    frames[3] = FrameSignal(
        1.5,
        green_field_ratio=0.73,
        motion_score=0.78,
        coordinated_motion_score=0.05,
        scorebug_visible=True,
    )
    frames[4] = FrameSignal(
        2.0,
        green_field_ratio=0.70,
        motion_score=0.65,
        coordinated_motion_score=0.05,
        scorebug_visible=True,
    )
    frames[5] = FrameSignal(
        2.5,
        green_field_ratio=0.68,
        motion_score=0.52,
        coordinated_motion_score=0.05,
        scorebug_visible=True,
    )
    frames[6] = FrameSignal(
        3.0,
        green_field_ratio=0.65,
        motion_score=0.35,
        coordinated_motion_score=0.05,
        scorebug_visible=True,
    )
    result = validate_live_play(frames, accepted_clock(), candidate_timestamp=1.8)

    assert not result.accepted
    assert result.snap_timestamp is None
    assert any("no coordinated motion/snap" in reason for reason in result.reasons)


def test_scorebug_ocr_is_used_when_frame_visibility_is_unknown() -> None:
    frames = [
        FrameSignal(0.0, green_field_ratio=0.75, motion_score=0.08),
        FrameSignal(0.5, green_field_ratio=0.74, motion_score=0.10),
        FrameSignal(0.75, green_field_ratio=0.74, motion_score=0.10),
        FrameSignal(1.0, green_field_ratio=0.73, motion_score=0.76, coordinated_motion_score=0.80),
        FrameSignal(1.5, green_field_ratio=0.70, motion_score=0.50, coordinated_motion_score=0.60),
        FrameSignal(2.0, green_field_ratio=0.65, motion_score=0.06),
    ]
    readings = [
        ScorebugReading(0.0, clock_seconds=300, quarter=2),
        ScorebugReading(0.5, clock_seconds=300, quarter=2),
        ScorebugReading(0.75, clock_seconds=299.5, quarter=2),
        ScorebugReading(1.0, clock_seconds=299, quarter=2),
        ScorebugReading(1.5, clock_seconds=299, quarter=2),
        ScorebugReading(2.0, clock_seconds=298, quarter=2),
    ]
    result = validate_live_play(frames, readings, candidate_timestamp=1.0)

    assert result.accepted
    assert dict(result.evidence)["scorebug_coverage"] == pytest.approx(1.0)


def test_rejects_scorebug_dropout_longer_than_allowed() -> None:
    frames = accepted_frames()
    frames[4] = FrameSignal(
        2.0,
        green_field_ratio=0.70,
        motion_score=0.65,
        coordinated_motion_score=0.70,
        scorebug_visible=False,
    )
    frames[5] = FrameSignal(
        2.5,
        green_field_ratio=0.68,
        motion_score=0.52,
        coordinated_motion_score=0.55,
        scorebug_visible=False,
    )
    result = validate_live_play(frames, accepted_clock(), candidate_timestamp=1.8)

    assert not result.accepted
    assert any("scorebug coverage" in reason for reason in result.reasons)


def test_unsorted_inputs_are_deterministic_and_not_mutated() -> None:
    frames = accepted_frames()
    readings = accepted_clock()
    original_frames = list(frames)
    original_readings = list(readings)

    first = validate_live_play(
        list(reversed(frames)), list(reversed(readings)), candidate_timestamp=1.8
    )
    second = validate_live_play(
        list(reversed(frames)), list(reversed(readings)), candidate_timestamp=1.8
    )

    assert first == second
    assert frames == original_frames
    assert readings == original_readings


def test_threshold_validation_is_strict() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        LivePlayConfig(min_green_field_ratio=1.1)
    with pytest.raises(ValueError, match="cannot exceed"):
        LivePlayConfig(min_pre_snap_seconds=4.0, max_snap_search_seconds=2.0)
    with pytest.raises(ValueError, match="clock_seconds"):
        ScorebugReading(1.0, clock_seconds=901)


def test_local_opencv_sampler_injects_scorebug_ocr(tmp_path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    output = tmp_path / "synthetic-live.avi"
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (160, 90),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV MJPG writer is unavailable")
    try:
        for index in range(20):
            frame = np.zeros((90, 160, 3), dtype=np.uint8)
            frame[:, :] = (45, 125, 45)
            left = 20 + index * 3
            frame[35:55, left : left + 12] = (0, 0, 255)
            writer.write(frame)
    finally:
        writer.release()

    calls: list[tuple[tuple[int, ...], float]] = []

    def ocr(crop, timestamp: float) -> dict[str, object]:
        calls.append((crop.shape, timestamp))
        return {
            "visible": True,
            "quarter": 1,
            "clock_seconds": 600.0 - timestamp,
            "confidence": 0.95,
        }

    sampled = sample_video_around_candidate(
        output,
        0.8,
        window_before=0.4,
        window_after=0.6,
        sample_fps=5.0,
        scorebug_ocr=ocr,
    )

    assert sampled.frames
    assert len(sampled.frames) == len(sampled.scorebug_readings) == len(calls)
    assert sampled.width == 160
    assert sampled.height == 90
    assert all(frame.scorebug_visible is True for frame in sampled.frames)
    assert all(reading.quarter == 1 for reading in sampled.scorebug_readings)
