import math

import pytest

from football_highlights.transcript_events import (
    FasterWhisperTranscriptAdapter,
    TranscriptDependencyError,
    TranscriptEvent,
    TranscriptSegment,
    deduplicate_events,
    detect_transcript_events,
    normalize_transcript_segments,
)


def segment(start, end, text, confidence=None):
    return TranscriptSegment(start, end, text, confidence=confidence)


def test_segment_validation_and_serialization():
    item = segment(1, 2.5, "The quarterback throws", 0.8)

    assert item.duration == 1.5
    assert item.to_dict() == {
        "start": 1,
        "end": 2.5,
        "text": "The quarterback throws",
        "confidence": 0.8,
    }

    with pytest.raises(ValueError):
        segment(2, 1, "backwards")
    with pytest.raises(ValueError):
        segment(0, 1, "", 0.5)
    with pytest.raises(ValueError):
        segment(0, 1, "bad confidence", 1.1)


def test_mapping_inputs_are_sorted_and_normalized():
    result = normalize_transcript_segments(
        [
            {"start": 5, "end": 6, "text": "later", "score": 0.7},
            {"start": 1, "end": 2, "transcript": "first"},
        ]
    )

    assert [item.text for item in result] == ["first", "later"]
    assert result[1].confidence == 0.7


def test_detects_touchdown_and_split_commentary_window():
    events = detect_transcript_events(
        [
            segment(10, 10.8, "He takes the snap and rolls right."),
            segment(10.9, 11.5, "He throws to the end zone."),
            segment(11.6, 12.0, "Touchdown!"),
        ]
    )

    touchdowns = [event for event in events if event.event_type == "touchdown"]
    assert len(touchdowns) == 1
    assert touchdowns[0].start == 10
    assert touchdowns[0].confidence > 0.9
    assert touchdowns[0].outcome == "score"


def test_detects_interception_fumble_field_goal_and_blocked_punt():
    events = detect_transcript_events(
        [
            segment(10, 11, "The pass is intercepted at the 20."),
            segment(50, 51, "The running back fumbles and the defense recovers."),
            segment(90, 91, "The 42 yard field goal is good."),
            segment(140, 141, "The punt is blocked and recovered in the end zone."),
        ]
    )

    assert {event.event_type for event in events} == {
        "interception",
        "fumble",
        "field_goal",
        "blocked_kick",
    }
    assert next(event for event in events if event.event_type == "field_goal").outcome == "made"
    assert next(event for event in events if event.event_type == "blocked_kick").outcome == "punt"


@pytest.mark.parametrize(
    ("text", "event_type", "outcome"),
    [
        ("The kick is no good, wide right.", "field_goal", "missed"),
        ("Picked off! He may take it all the way for a pick six.", "interception", "turnover"),
        ("A strip-sack and the ball is loose.", "fumble", "turnover_or_recovery"),
    ],
)
def test_detects_non_made_or_alternate_event_language(text, event_type, outcome):
    events = detect_transcript_events([segment(2, 3, text)])

    event = next(event for event in events if event.event_type == event_type)
    assert event.outcome == outcome


@pytest.mark.parametrize(
    "text",
    [
        "The kick is good.",
        "The kick good from 47 yards.",
        "It is good!",
        "He splits the uprights.",
        "The ball goes through the uprights.",
    ],
)
def test_detects_common_made_field_goal_language(text):
    event = next(
        event
        for event in detect_transcript_events([segment(2, 3, text)])
        if event.event_type == "field_goal"
    )

    assert event.outcome == "made"


def test_detects_big_plays_and_extracts_yardage_without_kick_false_positive():
    events = detect_transcript_events(
        [
            segment(1, 2, "Complete to the receiver for 24 yards."),
            segment(5, 6, "He breaks free for 51 yards!"),
            segment(9, 10, "A 55 yard field goal is good."),
        ]
    )

    big_plays = [event for event in events if event.event_type == "big_play"]
    assert [(event.yards, event.outcome) for event in big_plays] == [
        (24, "explosive_gain"),
        (51, "explosive_gain"),
    ]
    assert all(event.confidence >= 0.78 for event in big_plays)


def test_detects_first_down_and_fourth_down_results():
    events = detect_transcript_events(
        [
            segment(1, 2, "He scrambles for twelve yards and that's a first down."),
            segment(10, 11, "On fourth-and-one, he dives forward and gets the first down."),
            segment(20, 21, "They go for it on fourth down and the defense stops them."),
            segment(30, 31, "Fourth down attempt, incomplete pass."),
        ]
    )

    first_down = next(event for event in events if event.event_type == "first_down")
    assert first_down.outcome == "conversion"

    fourth_down = [event for event in events if event.event_type == "fourth_down"]
    assert [(event.outcome, event.start) for event in fourth_down] == [
        ("conversion", 10),
        ("stop", 20),
        ("stop", 30),
    ]


def test_detects_recovered_fumble_as_turnover():
    event = next(
        event
        for event in detect_transcript_events(
            [segment(5, 6, "The receiver fumbles and the defense recovers.")]
        )
        if event.event_type == "fumble"
    )

    assert event.outcome == "turnover"
    assert "recovery" in event.reasons[0]


def test_rejects_negated_or_future_non_scoring_discussion():
    events = detect_transcript_events(
        [
            segment(1, 2, "That pass was almost intercepted."),
            segment(4, 5, "He didn't fumble the ball."),
            segment(7, 8, "They will need a big play on fourth down."),
            segment(10, 11, "They are looking to convert on fourth down."),
        ]
    )

    assert events == []


def test_parses_explosive_returns_and_excludes_kick_attempt_yardage():
    events = detect_transcript_events(
        [
            segment(1, 2, "The punt return goes for 31 yards."),
            segment(5, 6, "A 52-yard field goal attempt is no good."),
        ]
    )

    big_plays = [event for event in events if event.event_type == "big_play"]
    assert len(big_plays) == 1
    assert big_plays[0].yards == 31


def test_touchdown_can_also_be_a_big_play():
    events = detect_transcript_events([segment(1, 2, "Touchdown pass for 62 yards!")])

    assert {event.event_type for event in events} == {"touchdown", "big_play"}
    assert next(event for event in events if event.event_type == "big_play").yards == 62


def test_replay_repetition_keeps_live_evidence_and_merges_indices():
    events = detect_transcript_events(
        [
            segment(10, 11, "Touchdown pass to Adams for 35 yards."),
            segment(48, 49, "Let's take another look at the touchdown pass to Adams for 35 yards."),
        ]
    )

    touchdowns = [event for event in events if event.event_type == "touchdown"]
    assert len(touchdowns) == 1
    assert touchdowns[0].start == 10
    assert touchdowns[0].is_replay_reference is False
    assert touchdowns[0].segment_indices == (0, 1)
    assert touchdowns[0].confidence > 0.9


def test_two_distinct_nearby_scores_are_not_deduplicated():
    events = detect_transcript_events(
        [
            segment(10, 11, "Touchdown run by Jones."),
            segment(40, 41, "Touchdown pass to Smith."),
        ],
        dedupe_window_seconds=90,
    )

    assert [event.start for event in events if event.event_type == "touchdown"] == [10, 40]


def test_replay_only_event_is_marked_when_no_live_description_exists():
    events = detect_transcript_events(
        [segment(20, 21, "Let's take another look at the touchdown.")]
    )

    event = next(event for event in events if event.event_type == "touchdown")
    assert event.is_replay_reference is True


def test_confidence_filter_and_transcript_confidence_tempering():
    events = detect_transcript_events(
        [segment(1, 2, "He finds the end zone.", confidence=0.2)],
        minimum_confidence=0.7,
    )
    assert events == []

    strong = detect_transcript_events(
        [segment(1, 2, "Touchdown!", confidence=0.2)],
        minimum_confidence=0.7,
    )
    assert len(strong) == 1
    assert strong[0].confidence == pytest.approx(0.98 * 0.8)


def test_deduplicate_events_prefers_non_replay_and_is_deterministic():
    replay = TranscriptEvent(
        "interception",
        80,
        81,
        0.99,
        "Replay: intercepted by Johnson",
        reasons=("interception",),
        segment_indices=(1,),
        is_replay_reference=True,
    )
    live = TranscriptEvent(
        "interception",
        20,
        21,
        0.85,
        "Pass intercepted by Johnson",
        reasons=("interception",),
        segment_indices=(0,),
    )

    result = deduplicate_events([replay, live])
    assert len(result) == 1
    assert result[0].start == 20
    assert result[0].is_replay_reference is False
    assert result[0].segment_indices == (0, 1)


def test_faster_whisper_adapter_is_lazy_and_uses_injected_model():
    class FakeSegment:
        start = 1
        end = 2
        text = "Touchdown!"
        avg_logprob = -0.1
        no_speech_prob = 0.05

    class FakeModel:
        def transcribe(self, path, **kwargs):
            assert path == "/tmp/game.mkv"
            assert kwargs["vad_filter"] is True
            return [FakeSegment()], object()

    adapter = FasterWhisperTranscriptAdapter(model=FakeModel())
    result = adapter.transcribe("/tmp/game.mkv")

    assert len(result) == 1
    assert result[0].text == "Touchdown!"
    assert result[0].confidence == pytest.approx(math.exp(-0.1) * 0.95)


def test_faster_whisper_missing_dependency_has_actionable_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def import_without_faster_whisper(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("not installed for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_faster_whisper)
    with pytest.raises(TranscriptDependencyError, match="faster-whisper is optional"):
        FasterWhisperTranscriptAdapter().transcribe("game.mkv")
