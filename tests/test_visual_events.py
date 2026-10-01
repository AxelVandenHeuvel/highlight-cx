from __future__ import annotations

import pytest

from football_highlights.visual_events import (
    EventCandidate,
    ScoreReading,
    VisualEventConfig,
    VisualObservation,
    detect_celebration_cuts,
    detect_replay_transitions,
    detect_score_changes,
    detect_sustained_field_action,
    detect_visual_events,
    fuse_candidates,
)


def test_score_change_requires_a_valid_monotonic_score_jump() -> None:
    readings = [
        ScoreReading(10.0, 0, 0, 0.95),
        ScoreReading(20.0, 7, 0, 0.96),
        # A one-frame OCR glitch is ignored as a backward score movement.
        ScoreReading(21.0, 0, 0, 0.99),
        ScoreReading(30.0, 7, 3, 0.94),
    ]

    candidates = detect_score_changes(readings)

    assert [(item.timestamp, item.score_delta) for item in candidates] == [
        (20.0, (7, 0)),
        (30.0, (0, 3)),
    ]
    assert "score-change:touchdown_or_extra_point" in candidates[0].reasons
    assert candidates[0].confidence == pytest.approx(0.9025)


def test_score_change_filters_low_confidence_and_deduplicates_readings() -> None:
    readings = [
        ScoreReading(0.0, 0, 0, 0.9),
        ScoreReading(4.0, 3, 0, 0.3),
        ScoreReading(5.0, 3, 0, 0.8),
        ScoreReading(5.1, 3, 0, 0.7),
    ]

    candidates = detect_score_changes(readings)

    assert len(candidates) == 1
    assert candidates[0].timestamp == 5.0
    assert candidates[0].score_delta == (3, 0)


def test_sustained_action_emits_one_candidate_for_a_contiguous_run() -> None:
    observations = [
        VisualObservation(0.0, field_action_score=0.7),
        VisualObservation(1.0, field_action_score=0.8),
        VisualObservation(2.0, field_action_score=0.75),
        VisualObservation(4.0, field_action_score=0.1),
    ]

    candidates = detect_sustained_field_action(observations)

    assert len(candidates) == 1
    assert candidates[0].kind == "sustained_field_action"
    assert candidates[0].source_start == 0.0
    assert candidates[0].source_end == 3.0
    assert candidates[0].confidence > 0.7


def test_replay_transition_uses_shot_cut_and_following_scorebug_loss() -> None:
    observations = [
        VisualObservation(10.0, field_action_score=0.7),
        VisualObservation(11.0, scene_change_score=0.9, field_action_score=0.2),
        VisualObservation(12.0, field_action_score=0.1, scorebug_visible=False),
        VisualObservation(13.0, field_action_score=0.1, scorebug_visible=False),
    ]

    candidates = detect_replay_transitions(observations)

    assert len(candidates) == 1
    assert candidates[0].kind == "replay_transition"
    assert "scorebug disappeared" in candidates[0].reasons
    assert candidates[0].confidence > 0.8


def test_celebration_cut_is_detected_from_post_cut_signal() -> None:
    observations = [
        VisualObservation(20.0, scene_change_score=0.8),
        VisualObservation(21.0, celebration_score=0.9),
        VisualObservation(22.0, celebration_score=0.7),
    ]

    candidates = detect_celebration_cuts(observations)

    assert len(candidates) == 1
    assert candidates[0].kind == "celebration_cut"
    assert candidates[0].timestamp == 20.0
    assert candidates[0].confidence == pytest.approx(0.845)


def test_fusion_is_deterministic_and_keeps_score_change_as_primary_event() -> None:
    score = EventCandidate(30.0, "score_change", 0.85, 25.0, 34.0, ("touchdown",), (7, 0))
    action = EventCandidate(29.0, "sustained_field_action", 0.70, 27.0, 32.0, ("motion",))
    replay = EventCandidate(35.0, "replay_transition", 0.80, 34.8, 37.0, ("cut",))

    first = fuse_candidates([replay, action, score])
    second = fuse_candidates([score, replay, action])

    assert first == second
    assert len(first) == 1
    assert first[0].kind == "score_change"
    assert first[0].score_delta == (7, 0)
    assert first[0].source_start == 25.0
    assert first[0].source_end == 37.0
    assert first[0].confidence > score.confidence


def test_combined_detection_returns_timestamped_fused_candidates() -> None:
    observations = [
        VisualObservation(40.0, field_action_score=0.8),
        VisualObservation(41.0, field_action_score=0.8),
        VisualObservation(42.0, field_action_score=0.8),
    ]
    readings = [ScoreReading(35.0, 0, 0, 0.95), ScoreReading(44.0, 7, 0, 0.95)]

    candidates = detect_visual_events(
        observations,
        readings,
        config=VisualEventConfig(score_change_pre_roll=2.0, score_change_post_roll=2.0),
    )

    assert len(candidates) == 1
    assert candidates[0].kind == "score_change"
    assert candidates[0].timestamp == 44.0
    assert candidates[0].confidence > 0.9


def test_invalid_candidate_and_config_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="source_end"):
        EventCandidate(1.0, "x", 0.5, 2.0, 2.0)
    with pytest.raises(ValueError, match="field_action_max_seconds"):
        VisualEventConfig(field_action_min_seconds=5.0, field_action_max_seconds=2.0)
