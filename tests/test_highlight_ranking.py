import pytest

from football_highlights.highlight_ranking import (
    NonScoringHighlightCandidate,
    deduplicate_non_scoring_candidates,
    normalize_non_scoring_candidates,
    rank_non_scoring_highlights,
)


def test_normalizes_aliases_and_clock_metadata() -> None:
    candidates = normalize_non_scoring_candidates(
        [
            {
                "type": "4th down conversion",
                "timestamp": 120.0,
                "text": "They convert on fourth down.",
                "confidence": 0.9,
                "quarter": "Q4",
                "clock": "1:42",
                "down": 4,
                "distance": 1,
            }
        ]
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.event_type == "fourth_down"
    assert candidate.quarter == 4
    assert candidate.clock_seconds == 102
    assert candidate.down == 4


def test_scoring_events_are_excluded_from_non_scoring_results() -> None:
    events = [
        {"event_type": "touchdown", "start": 1, "text": "Touchdown"},
        {"event_type": "field_goal", "start": 2, "text": "The kick is good"},
        {"event_type": "interception", "start": 3, "text": "Picked off"},
        {"event_type": "interception", "start": 4, "text": "Pick six touchdown"},
    ]

    result = normalize_non_scoring_candidates(events)

    assert [candidate.event_type for candidate in result] == ["interception"]


def test_semantic_and_late_game_context_change_ranking() -> None:
    events = [
        {
            "event_type": "big_play",
            "start": 30,
            "text": "A huge 45 yard gain down the sideline",
            "yards": 45,
            "confidence": 0.96,
        },
        {
            "event_type": "first_down",
            "start": 40,
            "text": "They convert on third and two for a first down",
            "confidence": 0.98,
            "quarter": 4,
            "clock": "0:48",
            "down": 3,
            "distance": 2,
        },
        {
            "event_type": "interception",
            "start": 50,
            "text": "Intercepted",
            "confidence": 0.98,
            "quarter": 2,
            "clock": "8:00",
        },
    ]

    result = rank_non_scoring_highlights(events)

    first_down = next(item for item in result if item.candidate.event_type == "first_down")
    big_play = next(item for item in result if item.candidate.event_type == "big_play")
    assert first_down.score > big_play.score
    assert any("under two minutes" in reason for reason in first_down.reasons)
    assert result[0].candidate.event_type == "interception"


def test_deduplication_prefers_live_event_over_replay() -> None:
    events = [
        {
            "event_type": "interception",
            "start": 100,
            "end": 104,
            "text": "Picked off by the safety near midfield",
            "confidence": 0.80,
        },
        {
            "event_type": "interception",
            "start": 158,
            "end": 162,
            "text": "Replay, picked off by the safety near midfield",
            "confidence": 0.99,
            "is_replay_reference": True,
        },
    ]

    result = deduplicate_non_scoring_candidates(events)

    assert len(result) == 1
    assert result[0].start == 100


def test_nearby_distinct_events_are_not_collapsed_when_descriptions_differ() -> None:
    events = [
        {
            "event_type": "big_play",
            "start": 100,
            "text": "A 35 yard pass to the left sideline",
            "yards": 35,
        },
        {
            "event_type": "big_play",
            "start": 125,
            "text": "A 28 yard run through the middle",
            "yards": 28,
        },
    ]

    result = deduplicate_non_scoring_candidates(events)

    assert [candidate.start for candidate in result] == [100, 125]


def test_play_id_deduplicates_even_when_commentary_differs() -> None:
    events = [
        NonScoringHighlightCandidate(
            "fumble", 100, 102, "Loose ball", confidence=0.75, play_id="p-12"
        ),
        NonScoringHighlightCandidate(
            "fumble", 135, 138, "Recovered by the defense", confidence=0.95, play_id="p-12"
        ),
    ]

    result = deduplicate_non_scoring_candidates(events)

    assert len(result) == 1
    assert result[0].start == 135


def test_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        normalize_non_scoring_candidates(
            [{"event_type": "punt", "start": 1, "text": "Punt"}]
        )
    with pytest.raises(ValueError, match="clock"):
        normalize_non_scoring_candidates(
            [{"event_type": "fumble", "start": 1, "text": "Fumble", "clock": "bad"}]
        )
    with pytest.raises(ValueError, match="max_results"):
        rank_non_scoring_highlights([], max_results=-1)
