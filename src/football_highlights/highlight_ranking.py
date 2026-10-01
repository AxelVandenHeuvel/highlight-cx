"""Semantic ranking and deduplication for non-scoring NFL highlights.

The functions in this module operate only on timestamped metadata extracted
from the local video's transcript.  They do not use a play-by-play feed,
network data, or manual timestamps.  The input may be a mapping (for example,
JSON emitted by a local analyzer) or any object with the same event
attributes as :class:`TranscriptEvent`.

The output is deliberately separate from the renderer.  A later video stage
can use ``candidate.start`` and ``candidate.end`` to find the live action,
while this module decides which non-scoring events deserve attention and
which repeated/replay descriptions represent one play.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

__all__ = [
    "NonScoringHighlightCandidate",
    "RankedHighlight",
    "deduplicate_non_scoring_candidates",
    "normalize_non_scoring_candidates",
    "rank_non_scoring_highlights",
]


_SUPPORTED_TYPES = {
    "interception",
    "fumble",
    "fourth_down",
    "first_down",
    "big_play",
}

_TYPE_ALIASES = {
    "pick": "interception",
    "picked_off": "interception",
    "forced_fumble": "fumble",
    "loose_ball": "fumble",
    "fourth-down": "fourth_down",
    "4th_down": "fourth_down",
    "4th_down_conversion": "fourth_down",
    "4th_down_stop": "fourth_down",
    "fourth_down_conversion": "fourth_down",
    "fourth_down_stop": "fourth_down",
    "crucial_first_down": "first_down",
    "first-down": "first_down",
    "explosive_play": "big_play",
    "chunk_play": "big_play",
    "long_gain": "big_play",
}

_SCORING_TYPES = {
    "touchdown",
    "field_goal",
    "extra_point",
    "safety",
    "score",
}

_REPLAY_WORDS = re.compile(
    r"\b(?:replay|take another look|let(?:'s| us) look again|we(?:'ll| will) show|"
    r"here it is again|earlier in the game|back to that play|watch this again)\b",
    re.IGNORECASE,
)

_STOP_WORDS = {
    "a",
    "after",
    "and",
    "at",
    "by",
    "for",
    "he",
    "his",
    "in",
    "is",
    "it",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_number(value: Any, *, name: str) -> float | None:
    return None if value is None else _finite_number(value, name=name)


def _first_value(source: Any, *names: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
        return default
    for name in names:
        value = getattr(source, name, None)
        if value is not None:
            return value
    return default


def _normalize_type(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("event_type must be a non-empty string")
    normalized = re.sub(r"\s+", "_", value.strip().lower())
    return _TYPE_ALIASES.get(normalized, normalized)


def _parse_quarter(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("quarter must be an integer, Q1-Q4, OT, or null")
    if isinstance(value, int):
        if 1 <= value <= 4:
            return value
        if value > 4:
            return "OT"
        raise ValueError("quarter must be positive")
    if isinstance(value, str):
        normalized = value.strip().upper().replace(" ", "")
        if normalized in {"OT", "OVERTIME"}:
            return "OT"
        match = re.fullmatch(r"Q?([1-4])(?:ST|ND|RD|TH)?", normalized)
        if match:
            return int(match.group(1))
    raise ValueError("quarter must be an integer, Q1-Q4, OT, or null")


def _parse_clock(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        match = re.fullmatch(r"(?:(\d+)\s*:\s*)?(\d+(?:\.\d+)?)", text)
        if match:
            minutes = float(match.group(1) or 0)
            seconds = float(match.group(2))
            result = minutes * 60 + seconds
        else:
            raise ValueError("clock must be seconds or an M:SS string")
    else:
        result = _finite_number(value, name="clock_seconds")
    if result < 0 or result > 15 * 60:
        raise ValueError("clock_seconds must be between 0 and 900")
    return result


def _text_tokens(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return {token for token in normalized.split() if token not in _STOP_WORDS}


def _text_similarity(left: str, right: str) -> float:
    left_tokens = _text_tokens(left)
    right_tokens = _text_tokens(right)
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, left.lower(), right.lower()).ratio()
    return max(jaccard, sequence)


@dataclass(frozen=True, slots=True)
class NonScoringHighlightCandidate:
    """Normalized event metadata used by the ranking stage.

    ``quarter`` and ``clock_seconds`` are optional because transcript-only
    recognition will not always read the broadcast scoreboard.  When present,
    they provide late-game context without becoming a hard requirement.
    """

    event_type: str
    start: float
    end: float
    text: str
    confidence: float = 0.5
    outcome: str = "unknown"
    yards: int | None = None
    quarter: int | str | None = None
    clock_seconds: float | None = None
    down: int | None = None
    distance: float | None = None
    score_margin: float | None = None
    is_replay_reference: bool = False
    play_id: str | None = None

    def __post_init__(self) -> None:
        if self.event_type not in _SUPPORTED_TYPES:
            raise ValueError(f"unsupported non-scoring event type: {self.event_type!r}")
        if not math.isfinite(self.start) or self.start < 0:
            raise ValueError("start must be finite and non-negative")
        if not math.isfinite(self.end) or self.end < self.start:
            raise ValueError("end must be finite and at least start")
        if not self.text.strip():
            raise ValueError("text must be non-empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.yards is not None and not isinstance(self.yards, int):
            raise ValueError("yards must be an integer or null")
        if self.down is not None and self.down not in {1, 2, 3, 4}:
            raise ValueError("down must be 1, 2, 3, or 4")
        if self.distance is not None and self.distance < 0:
            raise ValueError("distance must be non-negative")

    @classmethod
    def from_event(cls, event: Any) -> NonScoringHighlightCandidate:
        """Normalize a mapping or transcript-event-like object."""

        event_type = _normalize_type(_first_value(event, "event_type", "type", "kind"))
        start = _finite_number(_first_value(event, "start", "timestamp", "time"), name="start")
        raw_end = _first_value(event, "end", default=start + 1.0)
        text = str(_first_value(event, "text", "transcript", "description", default="event"))
        raw_yards = _first_value(event, "yards", "yardage")
        raw_down = _first_value(event, "down")
        raw_distance = _first_value(event, "distance", "to_go")
        raw_replay = _first_value(event, "is_replay_reference", "replay", default=False)
        play_id = _first_value(event, "play_id", "id")
        return cls(
            event_type=event_type,
            start=start,
            end=_finite_number(raw_end, name="end"),
            text=text,
            confidence=_finite_number(
                _first_value(event, "confidence", "score", default=0.5), name="confidence"
            ),
            outcome=str(_first_value(event, "outcome", "result", default="unknown")),
            yards=None if raw_yards is None else int(raw_yards),
            quarter=_parse_quarter(_first_value(event, "quarter", "period")),
            clock_seconds=_parse_clock(
                _first_value(
                    event,
                    "clock_seconds",
                    "clock_seconds_remaining",
                    "game_clock",
                    "clock",
                )
            ),
            down=None if raw_down is None else int(raw_down),
            distance=_optional_number(raw_distance, name="distance"),
            score_margin=_optional_number(
                _first_value(event, "score_margin", "score_differential"), name="score_margin"
            ),
            is_replay_reference=bool(raw_replay) or bool(_REPLAY_WORDS.search(text)),
            play_id=None if play_id is None else str(play_id),
        )


@dataclass(frozen=True, slots=True)
class RankedHighlight:
    """A candidate with an explainable, normalized priority score."""

    candidate: NonScoringHighlightCandidate
    score: float
    reasons: tuple[str, ...]


def normalize_non_scoring_candidates(
    events: Iterable[NonScoringHighlightCandidate | Mapping[str, Any] | Any],
) -> list[NonScoringHighlightCandidate]:
    """Normalize and retain only supported non-scoring event categories."""

    result: list[NonScoringHighlightCandidate] = []
    for event in events:
        raw_type = _normalize_type(_first_value(event, "event_type", "type", "kind"))
        # Scoring events are intentionally ignored rather than passed through
        # the non-scoring candidate validator.
        if raw_type in _SCORING_TYPES:
            continue
        candidate = (
            event
            if isinstance(event, NonScoringHighlightCandidate)
            else NonScoringHighlightCandidate.from_event(event)
        )
        if _is_scoring_candidate(candidate):
            continue
        result.append(candidate)
    return result


def _is_scoring_candidate(candidate: NonScoringHighlightCandidate) -> bool:
    if candidate.event_type in _SCORING_TYPES:
        return True
    text = candidate.text.lower()
    outcome = candidate.outcome.lower()
    return (
        "touchdown" in text
        or "field goal" in text
        or "extra point" in text
        or "pick six" in text
        or outcome in {"score", "scoring", "touchdown"}
    )


def _timing_adjustment(candidate: NonScoringHighlightCandidate) -> tuple[float, list[str]]:
    if candidate.quarter == "OT":
        return 0.14, ["overtime situation"]
    if candidate.quarter != 4:
        if candidate.clock_seconds is not None and candidate.clock_seconds <= 120:
            return 0.04, ["late-quarter situation"]
        return 0.0, []
    if candidate.clock_seconds is None:
        return 0.08, ["fourth-quarter situation"]
    if candidate.clock_seconds <= 120:
        return 0.18, ["fourth quarter under two minutes"]
    if candidate.clock_seconds <= 300:
        return 0.12, ["late fourth quarter"]
    return 0.08, ["fourth-quarter situation"]


def _semantic_adjustment(candidate: NonScoringHighlightCandidate) -> tuple[float, list[str]]:
    adjustment = 0.0
    reasons: list[str] = []
    outcome = candidate.outcome.lower()
    if candidate.event_type == "interception":
        adjustment += 0.05
        reasons.append("defensive turnover")
        if "return" in outcome or "turnover" in outcome:
            adjustment += 0.03
            reasons.append("return or possession change")
    elif candidate.event_type == "fumble":
        adjustment += 0.03
        reasons.append("loose-ball turnover evidence")
        if "recover" in outcome or "turnover" in outcome:
            adjustment += 0.04
            reasons.append("recovery or possession change")
    elif candidate.event_type == "fourth_down":
        adjustment += 0.07
        reasons.append("fourth-down result")
        if any(word in outcome for word in ("convert", "conversion", "stop", "turnover")):
            adjustment += 0.05
            reasons.append("fourth-down outcome identified")
    elif candidate.event_type == "first_down":
        if candidate.down in {3, 4}:
            adjustment += 0.10
            reasons.append(f"conversion on {candidate.down}th down")
        elif candidate.distance is not None and candidate.distance <= 2:
            adjustment += 0.07
            reasons.append("short-yardage first down")
        else:
            adjustment += 0.01
            reasons.append("first-down result")
    elif candidate.event_type == "big_play":
        if candidate.yards is not None and candidate.yards >= 40:
            adjustment += 0.18
            reasons.append(f"{candidate.yards}-yard explosive gain")
        elif candidate.yards is not None and candidate.yards >= 30:
            adjustment += 0.12
            reasons.append(f"{candidate.yards}-yard chunk play")
        elif candidate.yards is not None and candidate.yards >= 20:
            adjustment += 0.06
            reasons.append(f"{candidate.yards}-yard gain")
        else:
            adjustment += 0.01
            reasons.append("big-play commentary")
    return adjustment, reasons


def _score_candidate(candidate: NonScoringHighlightCandidate) -> RankedHighlight:
    base_scores = {
        "interception": 0.86,
        "fumble": 0.82,
        "fourth_down": 0.76,
        "first_down": 0.61,
        "big_play": 0.65,
    }
    score = base_scores[candidate.event_type]
    reasons = [candidate.event_type.replace("_", " ")]
    semantic, semantic_reasons = _semantic_adjustment(candidate)
    score += semantic
    reasons.extend(semantic_reasons)
    timing, timing_reasons = _timing_adjustment(candidate)
    score += timing
    reasons.extend(timing_reasons)
    if candidate.score_margin is not None and abs(candidate.score_margin) <= 8:
        score += 0.04
        reasons.append("close-score situation")
    confidence_factor = 0.65 + 0.35 * candidate.confidence
    score = min(1.0, max(0.0, score * confidence_factor))
    reasons.append(f"transcript confidence {candidate.confidence:.2f}")
    if candidate.is_replay_reference:
        score *= 0.82
        reasons.append("replay-reference penalty")
    return RankedHighlight(candidate=candidate, score=score, reasons=tuple(reasons))


def _same_play(left: NonScoringHighlightCandidate, right: NonScoringHighlightCandidate) -> bool:
    if left.play_id is not None and left.play_id == right.play_id:
        return True
    if left.event_type != right.event_type:
        return False
    gap = abs(left.start - right.start)
    if gap <= 12.0:
        return True
    if gap <= 90.0 and (left.is_replay_reference or right.is_replay_reference):
        return _text_similarity(left.text, right.text) >= 0.42
    return gap <= 90.0 and _text_similarity(left.text, right.text) >= 0.78


def deduplicate_non_scoring_candidates(
    events: Iterable[NonScoringHighlightCandidate | Mapping[str, Any] | Any],
    *,
    max_gap_seconds: float = 90.0,
) -> list[NonScoringHighlightCandidate]:
    """Collapse transcript windows and replay descriptions of one play.

    Live evidence wins over a replay reference.  Otherwise the candidate with
    higher confidence wins; ties favor the longer metadata window because it
    usually contains more of the play description.
    """

    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be non-negative")
    candidates = normalize_non_scoring_candidates(events)
    accepted: list[NonScoringHighlightCandidate] = []
    for candidate in sorted(candidates, key=lambda item: (item.start, item.end)):
        duplicate_index = next(
            (
                index
                for index in range(len(accepted) - 1, -1, -1)
                if abs(candidate.start - accepted[index].start) <= max_gap_seconds
                and _same_play(accepted[index], candidate)
            ),
            None,
        )
        if duplicate_index is None:
            accepted.append(candidate)
            continue
        previous = accepted[duplicate_index]
        candidate_key = (
            not candidate.is_replay_reference,
            candidate.confidence,
            candidate.end - candidate.start,
        )
        previous_key = (
            not previous.is_replay_reference,
            previous.confidence,
            previous.end - previous.start,
        )
        if candidate_key > previous_key:
            accepted[duplicate_index] = candidate
    return sorted(accepted, key=lambda item: item.start)


def rank_non_scoring_highlights(
    events: Iterable[NonScoringHighlightCandidate | Mapping[str, Any] | Any],
    *,
    max_results: int | None = None,
    dedupe_window_seconds: float = 90.0,
) -> list[RankedHighlight]:
    """Return explainably ranked non-scoring highlights, highest first."""

    if max_results is not None and max_results < 0:
        raise ValueError("max_results must be non-negative or null")
    deduplicated = deduplicate_non_scoring_candidates(
        events,
        max_gap_seconds=dedupe_window_seconds,
    )
    ranked = sorted(
        (_score_candidate(candidate) for candidate in deduplicated),
        key=lambda item: (-item.score, item.candidate.start),
    )
    return ranked if max_results is None else ranked[:max_results]
