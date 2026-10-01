"""Integration contracts for adding non-scoring highlights safely.

These tests deliberately use transcript fixtures and the pure live-play
validator instead of opening a real game file.  They document the boundary
the production pipeline must preserve when it expands beyond touchdowns and
field goals:

* transcript/audio evidence proposes a candidate; it never proves that the
  corresponding video interval is live action;
* replay-only evidence is discarded, and studio/commentary-looking video is
  rejected by the live-play gate;
* non-scoring events do not require a score delta, but they still require a
  validated live interval before becoming a manifest segment; and
* "first down"/"fourth down" language is only a candidate signal, not proof
  that the described play is a highlight.  It still needs play-context and
  live-video validation.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import football_highlights.video_only_analysis as analysis
from football_highlights.live_play import (
    FrameSignal,
    LivePlayResult,
    ScorebugReading,
    validate_live_play,
)
from football_highlights.media import VideoMetadata
from football_highlights.transcript_events import (
    TranscriptSegment,
    detect_transcript_events,
)


def _segment(start: float, text: str, *, end: float | None = None) -> TranscriptSegment:
    return TranscriptSegment(start, end if end is not None else start + 1.0, text, confidence=0.95)


def _accepted_live_frames() -> list[FrameSignal]:
    """A small live snap/play/settle sequence for validator contract tests."""

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


def _accepted_clock() -> list[ScorebugReading]:
    return [
        ScorebugReading(0.0, clock_seconds=600, quarter=1),
        ScorebugReading(1.5, clock_seconds=599, quarter=1),
        ScorebugReading(3.5, clock_seconds=598, quarter=1),
    ]


def test_non_scoring_transcript_vocabulary_produces_distinct_candidates() -> None:
    events = detect_transcript_events(
        [
            _segment(10, "The pass is intercepted at the 20."),
            _segment(30, "The running back fumbles and the defense recovers."),
            _segment(50, "The punt is blocked."),
            _segment(70, "Complete for 38 yards and a huge gain."),
        ]
    )

    by_type = {event.event_type: event for event in events}
    assert set(by_type) == {"interception", "fumble", "blocked_kick", "big_play"}
    assert by_type["interception"].outcome == "turnover"
    assert by_type["fumble"].outcome == "turnover"
    assert by_type["blocked_kick"].outcome == "punt"
    assert by_type["big_play"].yards == 38
    assert all(event.reasons for event in by_type.values())


def test_only_expanded_mode_routes_non_scoring_events_to_selection() -> None:
    interception = next(
        event
        for event in detect_transcript_events([_segment(10, "Picked off!")])
        if event.event_type == "interception"
    )

    assert not analysis._is_requested_event(interception, scoring_plays_only=True)
    assert analysis._is_requested_event(interception, scoring_plays_only=False)


@pytest.mark.parametrize("event_type", ["interception", "fumble", "blocked_kick", "big_play"])
def test_non_scoring_candidates_are_not_score_delta_gated(event_type: str) -> None:
    """A turnover/gain may be decisive without changing the scoreboard."""

    assert analysis._score_matches_event(event_type, None)
    assert analysis._score_matches_event(event_type, (0, 0))


def test_replay_only_transcript_evidence_is_not_a_live_candidate() -> None:
    events = detect_transcript_events(
        [_segment(20, "Let's take another look at the interception from earlier.")]
    )

    assert len(events) == 1
    assert events[0].is_replay_reference
    assert analysis._dedupe_semantic_events(events) == []


def test_live_validator_accepts_a_non_scoring_play_only_with_live_evidence() -> None:
    result = validate_live_play(_accepted_live_frames(), _accepted_clock(), candidate_timestamp=1.8)

    assert result.accepted
    assert result.live_start == pytest.approx(0.0)
    assert result.live_end == pytest.approx(4.0)
    assert "accepted: all strict live-play gates passed" in result.reasons


def test_live_validator_rejects_studio_or_commentary_like_candidate() -> None:
    frames = [
        FrameSignal(0.0, green_field_ratio=0.12, motion_score=0.08, scorebug_visible=False),
        FrameSignal(
            0.5,
            green_field_ratio=0.10,
            motion_score=0.72,
            coordinated_motion_score=0.80,
            scorebug_visible=False,
        ),
        FrameSignal(1.0, green_field_ratio=0.08, motion_score=0.25, scorebug_visible=False),
    ]
    result = validate_live_play(
        frames,
        [ScorebugReading(0.5, visible=False, clock_seconds=599, quarter=1)],
    )

    assert not result.accepted
    assert result.live_start is None
    assert any("no stable wide green-field" in reason for reason in result.reasons)


def test_down_language_is_only_a_candidate_until_live_validation() -> None:
    """Down language must not bypass the live-video gate."""

    events = detect_transcript_events(
        [
            _segment(10, "That was a huge moment on fourth down."),
            _segment(20, "They needed a first down to keep the drive alive."),
        ]
    )

    assert [event.event_type for event in events] == ["first_down"]
    assert not analysis._is_requested_event(events[0], scoring_plays_only=True)
    assert analysis._is_requested_event(events[0], scoring_plays_only=False)


def test_mocked_pipeline_requires_live_validation_before_manifest_segment(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expanded-mode candidate reaches the manifest only after the gate."""

    source = tmp_path / "game.mkv"
    cache = tmp_path / "transcript.json"
    cache.write_text(
        (
            '[{"start": 10.0, "end": 11.0, "text": "Pass intercepted by Johnson.", '
            '"confidence": 0.95}]'
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        analysis,
        "probe_video",
        lambda _source: VideoMetadata(120.0, 1920, 1080, 30.0, True),
    )
    monkeypatch.setattr(analysis, "detect_audio_events_from_video", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        analysis,
        "RapidOcrBackend",
        lambda: SimpleNamespace(
            read_score=lambda _crop: None,
            read_scorebug=lambda _crop, _timestamp: None,
        ),
    )
    monkeypatch.setattr(
        analysis,
        "sample_video_around_candidate",
        lambda *_args, **_kwargs: SimpleNamespace(frames=(), scorebug_readings=()),
    )

    live_calls: list[float] = []

    def accept_live(_frames, _readings, *, candidate_timestamp, config):
        live_calls.append(candidate_timestamp)
        return LivePlayResult(
            accepted=True,
            confidence=0.9,
            live_start=8.0,
            live_end=18.0,
            snap_timestamp=10.0,
            reasons=("accepted: test live evidence",),
        )

    monkeypatch.setattr(analysis, "validate_live_play", accept_live)
    monkeypatch.setattr(analysis, "_score_change_after_play", lambda *_args: None)
    monkeypatch.setattr(analysis, "_scan_goalpost_windows", lambda *_args, **_kwargs: [])

    manifest = analysis.analyze_video_only(
        source,
        max_highlights=4,
        scoring_plays_only=False,
        transcript_cache=cache,
    )

    assert live_calls == [11.0]
    assert len(manifest.segments) == 1
    segment = manifest.segments[0]
    assert segment.event_type == "interception"
    assert segment.source_start == pytest.approx(8.0)
    assert segment.source_end == pytest.approx(18.75)
    assert "strict live-play validation passed" in segment.reasons
