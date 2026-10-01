import cv2
import numpy as np

from football_highlights.transcript_events import TranscriptEvent, TranscriptSegment
from football_highlights.video_only_analysis import (
    _dedupe_semantic_events,
    _goalpost_visible,
    _infer_game_teams,
    _is_requested_event,
    _mentions_other_game,
    _score_matches_event,
)


def event(kind: str, start: float, *, reasons: tuple[str, ...], replay: bool = False):
    return TranscriptEvent(
        event_type=kind,
        start=start,
        end=start + 1,
        confidence=0.9,
        text=kind,
        outcome="test",
        reasons=reasons,
        segment_indices=(int(start),),
        is_replay_reference=replay,
    )


def test_dedupe_semantic_events_removes_replays_and_goal_line_false_positive() -> None:
    result = _dedupe_semantic_events(
        [
            event("touchdown", 10, reasons=("end-zone scoring language",)),
            event("touchdown", 20, reasons=("touchdown",)),
            event("touchdown", 50, reasons=("touchdown",), replay=True),
            event("touchdown", 70, reasons=("touchdown",)),
            event("touchdown", 140, reasons=("touchdown",)),
        ]
    )
    assert [item.start for item in result] == [20, 70, 140]


def test_other_game_mentions_are_rejected_from_inferred_matchup() -> None:
    transcript = [
        TranscriptSegment(0, 1, "Packers against the Jets"),
        TranscriptSegment(2, 3, "The Packers offense and Jets defense"),
    ]
    teams = _infer_game_teams(transcript)
    promo = TranscriptEvent(
        "interception",
        10,
        11,
        0.9,
        "Titans highlights after an interception",
        "turnover",
    )
    assert teams == {"packers", "jets"}
    assert _mentions_other_game(promo, teams)


def test_scoring_play_filter_keeps_touchdowns_and_field_goals() -> None:
    touchdown = event("touchdown", 10, reasons=("touchdown",))
    field_goal = event("field_goal", 20, reasons=("field goal made",))
    interception = event("interception", 30, reasons=("interception",))

    assert _is_requested_event(touchdown, True)
    assert _is_requested_event(field_goal, True)
    assert not _is_requested_event(interception, True)
    assert _is_requested_event(interception, False)


def test_scoring_candidates_require_the_expected_score_change() -> None:
    assert _score_matches_event("touchdown", (7, 0))
    assert _score_matches_event("touchdown", (0, 6))
    assert not _score_matches_event("touchdown", (3, 0))
    assert _score_matches_event("field_goal", (0, 3))
    assert not _score_matches_event("field_goal", (1, 0))
    assert not _score_matches_event("field_goal", None)


def test_goalpost_detector_requires_two_tall_yellow_uprights() -> None:
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    cv2.rectangle(frame, (220, 80), (235, 350), (0, 255, 255), -1)
    cv2.rectangle(frame, (500, 80), (515, 350), (0, 255, 255), -1)
    assert _goalpost_visible(frame)

    one_upright = np.zeros_like(frame)
    cv2.rectangle(one_upright, (220, 80), (235, 350), (0, 255, 255), -1)
    assert not _goalpost_visible(one_upright)
