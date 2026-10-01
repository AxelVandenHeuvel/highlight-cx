from __future__ import annotations

import pytest

from football_highlights.replay_detection import (
    FrameSignals,
    ReplayDetectionConfig,
    analyze_sequence,
    is_replay_or_non_live,
    repeated_fingerprint_ratio,
    slow_motion_cadence_proxy,
    studio_low_green_ratio,
)


def frame(timestamp: float, **changes: object) -> FrameSignals:
    values = {"timestamp": timestamp, "scorebug_visible": True}
    values.update(changes)
    return FrameSignals(**values)


def test_replay_wipe_and_logo_are_rejected_with_explainable_reasons() -> None:
    frames = [
        frame(0.0, scorebug_visible=False, replay_logo_score=0.9, wipe_score=0.8),
        frame(0.5, scorebug_visible=False, replay_logo_score=0.9, wipe_score=0.8),
        frame(1.0, scorebug_visible=False, replay_logo_score=0.8, wipe_score=0.7),
        frame(1.5, scorebug_visible=False, replay_logo_score=0.8, wipe_score=0.7),
    ]

    decision = analyze_sequence(frames)

    assert decision.rejected
    assert decision.confidence > 0.6
    assert any("replay/logo" in reason for reason in decision.reasons)
    assert any("wipe" in reason for reason in decision.reasons)
    assert ("replay_logo", 1.0) in decision.evidence


def test_studio_face_shot_and_scorebug_absence_are_rejected() -> None:
    frames = [
        frame(
            10.0,
            scorebug_visible=False,
            face_ratio=0.30,
            green_field_ratio=0.08,
            abrupt_cut_score=0.8,
        ),
        frame(
            10.5,
            scorebug_visible=False,
            face_ratio=0.28,
            green_field_ratio=0.10,
        ),
        frame(
            11.0,
            scorebug_visible=False,
            face_ratio=0.26,
            green_field_ratio=0.06,
        ),
    ]

    decision = analyze_sequence(frames)

    assert decision.rejected
    assert any("face-led" in reason for reason in decision.reasons)
    assert any("coincide" in reason for reason in decision.reasons)


def test_live_field_sequence_is_not_rejected() -> None:
    frames = [
        frame(20.0, motion_score=0.75, frame_delta_score=0.70, green_field_ratio=0.75),
        frame(20.5, motion_score=0.90, frame_delta_score=0.85, green_field_ratio=0.78),
        frame(21.0, motion_score=0.72, frame_delta_score=0.66, green_field_ratio=0.70),
        frame(21.5, motion_score=0.80, frame_delta_score=0.76, green_field_ratio=0.74),
    ]

    decision = analyze_sequence(frames)

    assert not decision.rejected
    assert decision.confidence < 0.2
    assert is_replay_or_non_live(frames) is False
    assert decision.reasons == ("visual sequence remains consistent with live field action",)


def test_scorebug_unknown_samples_are_not_counted_as_absent() -> None:
    frames = [
        frame(30.0, scorebug_visible=None),
        frame(30.5, scorebug_visible=True),
        frame(31.0, scorebug_visible=None),
    ]

    decision = analyze_sequence(frames)

    assert dict(decision.evidence)["scorebug_absence"] == 0.0
    assert not any("scorebug absent" in reason for reason in decision.reasons)


def test_repeated_frames_ignore_adjacent_duplicates_but_find_non_adjacent_repeats() -> None:
    adjacent = [
        frame(0.0, frame_fingerprint="a"),
        frame(0.5, frame_fingerprint="a"),
        frame(1.0, frame_fingerprint="b"),
    ]
    repeated = [
        frame(0.0, frame_fingerprint="a"),
        frame(0.5, frame_fingerprint="b"),
        frame(1.0, frame_fingerprint="a"),
        frame(1.5, frame_fingerprint="c"),
    ]

    assert repeated_fingerprint_ratio(adjacent) == 0.0
    assert repeated_fingerprint_ratio(repeated) == pytest.approx(0.25)


def test_slow_motion_proxy_uses_cadence_and_near_duplicate_signals() -> None:
    frames = [
        frame(0.0, motion_score=0.1, frame_delta_score=0.05),
        frame(0.25, motion_score=0.2, frame_delta_score=0.10),
        frame(0.5, motion_score=0.15, frame_delta_score=0.08),
        frame(0.75, motion_score=0.25, frame_delta_score=0.12),
    ]

    assert slow_motion_cadence_proxy(frames) == pytest.approx(1.0)


def test_studio_metric_and_custom_thresholds_are_deterministic() -> None:
    config = ReplayDetectionConfig(face_ratio_threshold=0.25, studio_green_max=0.20)
    frames = [
        frame(0.0, face_ratio=0.26, green_field_ratio=0.15),
        frame(0.5, face_ratio=0.12, green_field_ratio=0.10),
    ]

    assert studio_low_green_ratio(frames, config=config) == pytest.approx(0.5)


def test_invalid_signal_and_config_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="replay_logo_score"):
        FrameSignals(0.0, replay_logo_score=1.1)
    with pytest.raises(ValueError, match="rejection_threshold"):
        ReplayDetectionConfig(rejection_threshold=-0.1)


def test_empty_and_single_frame_sequences_have_safe_decisions() -> None:
    empty = analyze_sequence([])
    single = analyze_sequence([frame(4.0)], config=ReplayDetectionConfig(minimum_frames=2))

    assert not empty.rejected
    assert empty.confidence == 0.0
    assert not single.rejected
    assert single.start_timestamp == 4.0


def test_pure_api_does_not_require_opencv_or_a_video_file() -> None:
    frames = [frame(5.0), frame(5.5, motion_score=0.8, green_field_ratio=0.8)]

    decision = analyze_sequence(tuple(frames))

    assert decision.start_timestamp == 5.0
    assert decision.end_timestamp == 5.5
