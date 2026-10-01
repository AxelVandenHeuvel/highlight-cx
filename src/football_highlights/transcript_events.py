"""Detect football highlights from a local broadcast transcript.

This module intentionally does not depend on a play-by-play feed.  It turns
timestamped commentary into explainable event candidates using deterministic
patterns, then removes the repeated descriptions that are common when a
broadcast shows a replay.  The optional faster-whisper adapter is lazy: the
rest of the module remains importable and testable without installing a speech
recognition model.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Protocol

__all__ = [
    "FasterWhisperTranscriptAdapter",
    "TranscriptDependencyError",
    "TranscriptEvent",
    "TranscriptSegment",
    "TranscriptTranscriber",
    "deduplicate_events",
    "detect_events",
    "detect_transcript_events",
    "normalize_transcript_segments",
]


class TranscriptDependencyError(RuntimeError):
    """Raised when the optional local transcription dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One timestamped segment returned by a local speech recognizer."""

    start: float
    end: float
    text: str
    confidence: float | None = None
    speaker: str | None = None
    language: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.start, bool) or not isinstance(self.start, (int, float)):
            raise ValueError("segment start must be numeric")
        if isinstance(self.end, bool) or not isinstance(self.end, (int, float)):
            raise ValueError("segment end must be numeric")
        if not math.isfinite(float(self.start)) or self.start < 0:
            raise ValueError("segment start must be finite and non-negative")
        if not math.isfinite(float(self.end)) or self.end < self.start:
            raise ValueError("segment end must be finite and at least start")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("segment text must be a non-empty string")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, (int, float))
            or not math.isfinite(float(self.confidence))
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("segment confidence must be between 0 and 1")

    @property
    def duration(self) -> float:
        """Return the segment duration in seconds."""

        return float(self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        result: dict[str, Any] = {
            "start": self.start,
            "end": self.end,
            "text": self.text,
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.speaker is not None:
            result["speaker"] = self.speaker
        if self.language is not None:
            result["language"] = self.language
        return result


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    """An automatically detected football event.

    ``start`` and ``end`` describe the transcript evidence, not the final
    video edit boundaries.  The boundary-refinement stage can use these times
    as a search window.  ``is_replay_reference`` identifies evidence that was
    spoken during a replay; deduplication normally keeps the earlier live-play
    evidence instead.
    """

    event_type: str
    start: float
    end: float
    confidence: float
    text: str
    outcome: str = "unknown"
    yards: int | None = None
    reasons: tuple[str, ...] = ()
    segment_indices: tuple[int, ...] = ()
    is_replay_reference: bool = False

    def __post_init__(self) -> None:
        allowed_types = {
            "touchdown",
            "interception",
            "fumble",
            "field_goal",
            "blocked_kick",
            "big_play",
            "first_down",
            "fourth_down",
        }
        if self.event_type not in allowed_types:
            raise ValueError(f"unsupported transcript event type: {self.event_type!r}")
        if self.start < 0 or self.end < self.start:
            raise ValueError("event timestamps must be non-negative and ordered")
        if not 0 <= self.confidence <= 1:
            raise ValueError("event confidence must be between 0 and 1")
        if not self.text.strip():
            raise ValueError("event text must be non-empty")
        if self.yards is not None and not isinstance(self.yards, int):
            raise ValueError("event yards must be an integer or null")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation suitable for an audit file."""

        return {
            "event_type": self.event_type,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "text": self.text,
            "outcome": self.outcome,
            "yards": self.yards,
            "reasons": list(self.reasons),
            "segment_indices": list(self.segment_indices),
            "is_replay_reference": self.is_replay_reference,
        }


class TranscriptTranscriber(Protocol):
    """Protocol implemented by local transcript providers."""

    def transcribe(self, audio_or_video: str | Path) -> list[TranscriptSegment]:
        """Return timestamped transcript segments."""


class FasterWhisperTranscriptAdapter:
    """Lazy adapter around the optional local ``faster-whisper`` package.

    The model is loaded only when :meth:`transcribe` is called.  No network
    request is made by this adapter; the model must already be available in
    the local faster-whisper cache or supplied through ``model``.
    """

    def __init__(
        self,
        model_size: str = "small.en",
        *,
        device: str = "cpu",
        compute_type: str = "int8",
        model: Any | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = model

    def _model_or_load(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptDependencyError(
                "faster-whisper is optional; install it locally to transcribe video"
            ) from exc
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        return self._model

    def transcribe(self, audio_or_video: str | Path) -> list[TranscriptSegment]:
        """Transcribe a local media file into normalized segments."""

        model = self._model_or_load()
        segments, _info = model.transcribe(
            str(audio_or_video),
            beam_size=5,
            vad_filter=True,
        )
        result: list[TranscriptSegment] = []
        for segment in segments:
            text = str(getattr(segment, "text", "")).strip()
            if not text:
                continue
            average_log_probability = getattr(segment, "avg_logprob", None)
            no_speech_probability = getattr(segment, "no_speech_prob", 0.0)
            confidence = _whisper_confidence(average_log_probability, no_speech_probability)
            result.append(
                TranscriptSegment(
                    start=float(segment.start),
                    end=float(segment.end),
                    text=text,
                    confidence=confidence,
                )
            )
        return result


def _whisper_confidence(avg_logprob: Any, no_speech_prob: Any) -> float | None:
    if not isinstance(avg_logprob, (int, float)) or not math.isfinite(float(avg_logprob)):
        return None
    no_speech = (
        float(no_speech_prob)
        if isinstance(no_speech_prob, (int, float)) and math.isfinite(float(no_speech_prob))
        else 0.0
    )
    no_speech = min(1.0, max(0.0, no_speech))
    # Whisper log probabilities are usually negative.  Exponentiating gives a
    # useful bounded quality signal without pretending it is calibrated odds.
    return min(1.0, max(0.0, math.exp(min(0.0, float(avg_logprob))) * (1 - no_speech)))


def normalize_transcript_segments(
    segments: Iterable[TranscriptSegment | Mapping[str, Any]],
) -> list[TranscriptSegment]:
    """Coerce, validate, and sort transcript records by start time."""

    normalized: list[TranscriptSegment] = []
    for segment in segments:
        if isinstance(segment, TranscriptSegment):
            normalized.append(segment)
            continue
        if not isinstance(segment, Mapping):
            raise TypeError("transcript segments must be TranscriptSegment or mapping values")
        text = segment.get("text", segment.get("transcript", ""))
        confidence = segment.get("confidence", segment.get("score"))
        normalized.append(
            TranscriptSegment(
                start=float(segment["start"]),
                end=float(segment.get("end", segment["start"])),
                text=str(text),
                confidence=None if confidence is None else float(confidence),
                speaker=segment.get("speaker"),
                language=segment.get("language"),
            )
        )
    return sorted(normalized, key=lambda item: (item.start, item.end))


_REPLAY_RE = re.compile(
    r"\b(?:replay|take another look|let(?:'s| us) look again|we(?:'ll| will) show|"
    r"here it is again|earlier in the game|back to that play|watch this again)\b",
    re.IGNORECASE,
)
_TOUCHDOWN_RE = re.compile(
    r"\b(?:touchdown|t\.?\s*d\.?|pick[- ]?six|returns? .*? for a score)\b",
    re.IGNORECASE,
)
_END_ZONE_RE = re.compile(
    r"\b(?:into|in(to)? the|reaches? the|finds? the) end zone\b|\bgoal[- ]line\b",
    re.IGNORECASE,
)
_INTERCEPTION_RE = re.compile(
    r"\b(?:intercept(?:ed|ion)|picked off|pick[- ]?six|gets? the pick)\b", re.IGNORECASE
)
_INTERCEPTION_NEGATION_RE = re.compile(
    r"\b(?:not|never|no|isn['’]?t|wasn['’]?t|was not|is not|almost|nearly|could have|"
    r"would have|might have)\b.{0,24}\b(?:intercept(?:ed|ion)|picked off|pick[- ]?six)\b|"
    r"\b(?:intercept(?:ed|ion)|picked off|pick[- ]?six)\b.{0,16}\b(?:no|not)\b",
    re.IGNORECASE,
)
_FUMBLE_RE = re.compile(
    r"\b(?:fumble[sd]?|muff(?:ed|s)?|loose ball|strip[- ]sack|stripped)\b|"
    r"\brecovered by (?:the )?(?:defense|defensive|[A-Z]{2,3}\b)",
    re.IGNORECASE,
)
_FUMBLE_NEGATION_RE = re.compile(
    r"\b(?:didn['’]?t|doesn['’]?t|not|never|no|isn['’]?t|wasn['’]?t|almost|nearly|"
    r"could have|would have|might have)\b.{0,24}\b(?:fumble|muff|loose|strip(?:ped)?|"
    r"recovered)\b|\b(?:fumble|muff|loose|strip(?:ped)?|recovered)\b.{0,16}\b(?:no|not)\b",
    re.IGNORECASE,
)
_KICK_BLOCK_RE = re.compile(
    r"\bblocked\b.{0,45}\b(?:punt|field goal|extra point|kick|kickoff)\b|"
    r"\b(?:punt|field goal|extra point|kick|kickoff)\b.{0,45}\bblocked\b",
    re.IGNORECASE,
)
_FIELD_GOAL_RE = re.compile(
    r"\bfield goal\b|"
    r"\b(?:the )?kick\s+(?:is\s+)?(?:good|no good|missed)\b|"
    r"\b(?:it|that)\s+is\s+(?:good|no good)\b|"
    r"\b(?:splits?|through)\s+the\s+uprights\b",
    re.IGNORECASE,
)
_MADE_KICK_RE = re.compile(
    r"\b(?:good|makes?|made|successful|splits? the uprights|through the uprights)\b",
    re.IGNORECASE,
)
_MISSED_KICK_RE = re.compile(
    r"\b(?:no good|miss(?:ed|es)?|wide (?:left|right)|short|off the upright|doink)\b",
    re.IGNORECASE,
)
_YARDS_RE = re.compile(
    r"\b(?:for|gain(?:s|ed)? of|goes for|picks up|runs? for|pass(?:es)? for)\s+"
    r"(?P<yards>\d{2,3})\s+yards?\b|\b(?P<standalone>\d{2,3})[- ]yards?\b",
    re.IGNORECASE,
)
_BIG_PLAY_RE = re.compile(
    r"\b(?:big play|explosive play|breaks? free|breaks? away|wide open deep|"
    r"goes? the distance|all the way)\b",
    re.IGNORECASE,
)
_BIG_PLAY_ACTION_RE = re.compile(
    r"\b(?:complete(?:d)?|catch(?:es|ed)?|run(?:s|ning)?|rush(?:es|ed)?|"
    r"pass(?:es|ed)?|throw(?:s|n)?|gain(?:s|ed)?|pick(?:s|ed)? up|"
    r"return(?:s|ed)?|takes? it|breaks? free|breaks? away|goes? for)\b",
    re.IGNORECASE,
)
_KICK_ATTEMPT_RE = re.compile(
    r"\b(?:field goal|extra point|punt)\b(?!\s+return)|\b(?:kick|punt)\s+"
    r"(?:is|was|goes|went)\s+(?:good|no good|away|out of bounds)\b",
    re.IGNORECASE,
)
_FIRST_DOWN_RE = re.compile(
    r"\b(?:first|1st)\s+down\b|\bmove(?:s|d)?\s+the chains\b", re.IGNORECASE
)
_FIRST_DOWN_SUCCESS_RE = re.compile(
    r"\b(?:gets?|got|picks? up|converts?|converted|moves?|moved|keeps?|kept|"
    r"enough for|that's|that is|first down)\b.{0,32}\b(?:first|1st)\s+down\b|"
    r"\b(?:first|1st)\s+down\b.{0,24}\b(?:gets?|got|converts?|converted|"
    r"moves?|moved|keeps?|kept|complete(?:d)?|runs?|gain(?:s|ed)?)\b|"
    r"\bmove(?:s|d)?\s+the chains\b",
    re.IGNORECASE,
)
_FOURTH_DOWN_ANCHOR = (
    r"\b(?:fourth|4th)(?:\s*-\s*(?:and[-\s]*)?(?:\d+|one|two|three|four|goal)|"
    r"\s+and\s+(?:\d+|one|two|three|four|goal)|\s+down)\b"
)
_FOURTH_DOWN_RE = re.compile(_FOURTH_DOWN_ANCHOR, re.IGNORECASE)
_FOURTH_DOWN_ACTION_RE = re.compile(
    r"\b(?:snap|snaps?|pass(?:es|ed)?|throw(?:s|n)?|run(?:s|ning)?|rush(?:es|ed)?|"
    r"dive(?:s|d)?|"
    r"go(?:es)? for it|tries?|attempt(?:s|ed)?|converts?|converted|complete(?:s|d)?|"
    r"incomplete|short|stopped|stop(?:ped)?|turnover on downs|fails?|miss(?:es|ed)?)\b",
    re.IGNORECASE,
)
_FOURTH_DOWN_CONVERSION_RE = re.compile(
    r"\b(?:converts?|converted|gets?|got|picks? up|moves?|moved|keeps?|kept|"
    r"successful|made it|first down)\b.{0,38}"
    + _FOURTH_DOWN_ANCHOR
    + r"|"
    + _FOURTH_DOWN_ANCHOR
    + r".{0,38}\b(?:converts?|converted|gets?|got|picks? up|moves?|moved|"
    r"keeps?|kept|successful|first down)\b",
    re.IGNORECASE,
)
_FOURTH_DOWN_STOP_RE = re.compile(
    r"\b(?:turnover on downs|stops?|stuffed|denied|fails?|failed|short of the line|"
    r"incomplete|incompletions?|doesn['’]?t get|didn['’]?t get|defense holds?)\b",
    re.IGNORECASE,
)
_FUTURE_OR_HYPOTHETICAL_RE = re.compile(
    r"\b(?:will|would|could|might|may|going to|gonna|plan(?:s|ning)? to|need(?:s)? to|"
    r"want(?:s)? to|hope(?:s)? to|looking to|trying to)\b.{0,28}\b(?:intercept|fumble|"
    r"first down|fourth down|big play|gain|convert)\b",
    re.IGNORECASE,
)


def _yards(text: str) -> int | None:
    matches = list(_YARDS_RE.finditer(text))
    if not matches:
        return None
    match = matches[-1]
    raw = match.group("yards") or match.group("standalone")
    return int(raw) if raw is not None else None


def _normalized_text(text: str) -> str:
    text = text.lower().replace("'", "")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _content_tokens(text: str) -> set[str]:
    stop_words = {
        "a",
        "and",
        "at",
        "by",
        "for",
        "he",
        "his",
        "is",
        "it",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
    return {token for token in _normalized_text(text).split() if token not in stop_words}


def _fumble_outcome(text: str) -> str:
    normalized = _normalized_text(text)
    if re.search(
        r"\b(?:turnover|lost|defen[cs]e recovered|recovered by the defense|"
        r"defen[cs]e recover(?:ed|s)?|recovered by [a-z]+)\b",
        normalized,
    ):
        return "turnover"
    if re.search(r"\b(?:recover(?:ed|s)?|recovery)\b", normalized):
        return "recovered_fumble"
    return "turnover_or_recovery"


def _has_live_first_down_action(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:snap(?:s|ped)?|run(?:s|ning)?|rush(?:es|ed)?|scramble(?:s|d)?|"
            r"dive(?:s|d)?|complete(?:s|d)?|catch(?:es|ed)?|pass(?:es|ed)?|"
            r"throw(?:s|n)?|gain(?:s|ed)?|picks? up|gets?|got|converts?|converted|"
            r"moves?|moved|keeps?|kept|enough for)\b",
            text,
            re.IGNORECASE,
        )
    )


def _is_future_or_hypothetical(text: str) -> bool:
    return bool(_FUTURE_OR_HYPOTHETICAL_RE.search(text))


def _text_similarity(left: str, right: str) -> float:
    left_normalized = _normalized_text(left)
    right_normalized = _normalized_text(right)
    if not left_normalized or not right_normalized:
        return 0.0
    ratio = SequenceMatcher(None, left_normalized, right_normalized).ratio()
    left_tokens = _content_tokens(left)
    right_tokens = _content_tokens(right)
    union = left_tokens | right_tokens
    jaccard = len(left_tokens & right_tokens) / len(union) if union else 0.0
    return max(ratio, jaccard)


def _windowed_segments(
    segments: Sequence[TranscriptSegment],
    *,
    max_join_gap: float,
    max_segments: int,
) -> list[tuple[str, float, float, tuple[int, ...]]]:
    windows: list[tuple[str, float, float, tuple[int, ...]]] = []
    for index, segment in enumerate(segments):
        windows.append((segment.text, segment.start, segment.end, (index,)))
        text_parts = [segment.text]
        indices = [index]
        end = segment.end
        for next_index in range(index + 1, min(len(segments), index + max_segments)):
            following = segments[next_index]
            if following.start - end > max_join_gap:
                break
            text_parts.append(following.text)
            indices.append(next_index)
            end = following.end
            windows.append((" ".join(text_parts), segment.start, end, tuple(indices)))
    return windows


def _confidence(segment_confidences: Sequence[float | None], base: float) -> float:
    available = [value for value in segment_confidences if value is not None]
    if not available:
        return base
    # Transcript quality should temper, but not erase, a strong football cue.
    return max(0.0, min(1.0, base * (0.75 + 0.25 * min(available))))


def _event_candidates(
    text: str,
    start: float,
    end: float,
    indices: tuple[int, ...],
    segment_confidences: Sequence[float | None],
) -> list[TranscriptEvent]:
    replay_reference = bool(_REPLAY_RE.search(text))
    yards = _yards(text)
    common = {
        "start": start,
        "end": end,
        "text": text.strip(),
        "yards": yards,
        "segment_indices": indices,
        "is_replay_reference": replay_reference,
    }
    candidates: list[TranscriptEvent] = []
    if _KICK_BLOCK_RE.search(text):
        kick_type = "punt" if re.search(r"\bpunt\b", text, re.I) else "kick"
        candidates.append(
            TranscriptEvent(
                event_type="blocked_kick",
                confidence=_confidence(segment_confidences, 0.97),
                outcome=kick_type,
                reasons=(f"blocked {kick_type}",),
                **common,
            )
        )
    if (
        _INTERCEPTION_RE.search(text)
        and not _INTERCEPTION_NEGATION_RE.search(text)
        and not _is_future_or_hypothetical(text)
    ):
        candidates.append(
            TranscriptEvent(
                event_type="interception",
                confidence=_confidence(segment_confidences, 0.98),
                outcome="turnover",
                reasons=("interception",),
                **common,
            )
        )
    if (
        _FUMBLE_RE.search(text)
        and not _FUMBLE_NEGATION_RE.search(text)
        and not _is_future_or_hypothetical(text)
    ):
        candidates.append(
            TranscriptEvent(
                event_type="fumble",
                confidence=_confidence(segment_confidences, 0.94),
                outcome=_fumble_outcome(text),
                reasons=(
                    "fumble recovery/turnover"
                    if _fumble_outcome(text) != "turnover_or_recovery"
                    else "fumble or loose ball",
                ),
                **common,
            )
        )
    touchdown_match = _TOUCHDOWN_RE.search(text)
    end_zone_match = _END_ZONE_RE.search(text)
    # Broadcast commentary can say that a blocked punt was recovered in the
    # end zone without describing a touchdown.  Require explicit scoring
    # language for kick plays, turnovers, and loose-ball recoveries.
    inferred_end_zone_score = end_zone_match and not (
        _KICK_BLOCK_RE.search(text)
        or re.search(r"\b(?:punt|kickoff|fumble|loose ball|recovered)\b", text, re.I)
    )
    if touchdown_match or inferred_end_zone_score:
        reason = "touchdown" if touchdown_match else "end-zone scoring language"
        base = 0.98 if touchdown_match else 0.70
        candidates.append(
            TranscriptEvent(
                event_type="touchdown",
                confidence=_confidence(segment_confidences, base),
                outcome="score",
                reasons=(reason,),
                **common,
            )
        )
    if _FIELD_GOAL_RE.search(text) and not _KICK_BLOCK_RE.search(text):
        if _MISSED_KICK_RE.search(text):
            outcome = "missed"
        elif _MADE_KICK_RE.search(text):
            outcome = "made"
        else:
            outcome = "attempt"
        reason = "field goal made" if outcome == "made" else "field goal attempt"
        candidates.append(
            TranscriptEvent(
                event_type="field_goal",
                confidence=_confidence(segment_confidences, 0.96 if outcome != "attempt" else 0.82),
                outcome=outcome,
                reasons=(reason,),
                **common,
            )
        )
    fourth_down_match = _FOURTH_DOWN_RE.search(text)
    fourth_down_action = bool(_FOURTH_DOWN_ACTION_RE.search(text))
    if fourth_down_match and fourth_down_action and not _is_future_or_hypothetical(text):
        if _FOURTH_DOWN_STOP_RE.search(text):
            fourth_down_outcome = "stop"
            fourth_down_reason = "fourth-down stop or turnover on downs"
        elif _FOURTH_DOWN_CONVERSION_RE.search(text):
            fourth_down_outcome = "conversion"
            fourth_down_reason = "fourth-down conversion"
        else:
            fourth_down_outcome = "attempt"
            fourth_down_reason = "fourth-down attempt"
        candidates.append(
            TranscriptEvent(
                event_type="fourth_down",
                confidence=_confidence(
                    segment_confidences,
                    0.94 if fourth_down_outcome != "attempt" else 0.78,
                ),
                outcome=fourth_down_outcome,
                reasons=(fourth_down_reason,),
                **common,
            )
        )
    first_down_success = bool(_FIRST_DOWN_SUCCESS_RE.search(text)) or (
        bool(_FIRST_DOWN_RE.search(text)) and _has_live_first_down_action(text)
    )
    if (
        first_down_success
        and not fourth_down_match
        and not _is_future_or_hypothetical(text)
        and not _FOURTH_DOWN_STOP_RE.search(text)
    ):
        candidates.append(
            TranscriptEvent(
                event_type="first_down",
                confidence=_confidence(segment_confidences, 0.86),
                outcome="conversion",
                reasons=("successful first down",),
                **common,
            )
        )
    yard_matches = list(_YARDS_RE.finditer(text))
    generic_big_play = bool(_BIG_PLAY_RE.search(text))
    yard_big_play = yards is not None and yards >= 20
    # A joined transcript window can cover two consecutive plays.  Do not
    # turn its last yardage mention into a misleading combined event; the
    # individual segments will each be evaluated separately.
    single_yardage_evidence = len(yard_matches) <= 1
    if (
        (
            generic_big_play
            or (
                yard_big_play
                and single_yardage_evidence
                and _BIG_PLAY_ACTION_RE.search(text)
            )
        )
        and not _is_future_or_hypothetical(text)
        and not _KICK_ATTEMPT_RE.search(text)
    ):
        if yards is not None and yards >= 40:
            base = 0.91
            reason = f"{yards}-yard explosive play"
        elif yard_big_play:
            base = 0.78
            reason = f"{yards}-yard big play"
        else:
            base = 0.64
            reason = "commentary calls it a big or explosive play"
        candidates.append(
            TranscriptEvent(
                event_type="big_play",
                confidence=_confidence(segment_confidences, base),
                outcome="explosive_gain",
                reasons=(reason,),
                **common,
            )
        )
    return candidates


def _is_duplicate(left: TranscriptEvent, right: TranscriptEvent, max_gap: float) -> bool:
    if left.event_type != right.event_type:
        return False
    gap = abs(right.start - left.start)
    if gap > max_gap:
        return False
    if set(left.segment_indices) & set(right.segment_indices):
        return True
    similarity = _text_similarity(left.text, right.text)
    if similarity >= 0.76:
        return True
    if left.is_replay_reference or right.is_replay_reference:
        shared = _content_tokens(left.text) & _content_tokens(right.text)
        return similarity >= 0.48 and len(shared) >= 2
    # A bare "touchdown" can describe two nearby scores, so only collapse it
    # when the transcript windows are nearly coincident.
    return False


def _prefer_event(left: TranscriptEvent, right: TranscriptEvent) -> TranscriptEvent:
    left_key = (not left.is_replay_reference, left.confidence, -left.start)
    right_key = (not right.is_replay_reference, right.confidence, -right.start)
    return left if left_key >= right_key else right


def _merge_duplicate(left: TranscriptEvent, right: TranscriptEvent) -> TranscriptEvent:
    preferred = _prefer_event(left, right)
    reasons = tuple(dict.fromkeys((*left.reasons, *right.reasons)))
    indices = tuple(sorted(set(left.segment_indices) | set(right.segment_indices)))
    return TranscriptEvent(
        event_type=preferred.event_type,
        start=preferred.start,
        end=preferred.end,
        confidence=max(left.confidence, right.confidence),
        text=preferred.text,
        outcome=preferred.outcome,
        yards=preferred.yards if preferred.yards is not None else left.yards or right.yards,
        reasons=reasons,
        segment_indices=indices,
        is_replay_reference=left.is_replay_reference and right.is_replay_reference,
    )


def deduplicate_events(
    events: Iterable[TranscriptEvent],
    *,
    max_gap_seconds: float = 90.0,
) -> list[TranscriptEvent]:
    """Collapse transcript-window and replay repetitions deterministically."""

    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds must be non-negative")
    result: list[TranscriptEvent] = []
    for event in sorted(events, key=lambda item: (item.start, item.end, item.event_type)):
        duplicate_index = next(
            (
                index
                for index in range(len(result) - 1, -1, -1)
                if _is_duplicate(result[index], event, max_gap_seconds)
            ),
            None,
        )
        if duplicate_index is None:
            result.append(event)
        else:
            result[duplicate_index] = _merge_duplicate(result[duplicate_index], event)
    return result


def detect_transcript_events(
    segments: Iterable[TranscriptSegment | Mapping[str, Any]],
    *,
    max_join_gap_seconds: float = 1.0,
    max_window_segments: int = 3,
    dedupe_window_seconds: float = 90.0,
    minimum_confidence: float = 0.0,
) -> list[TranscriptEvent]:
    """Detect and deduplicate football events from timestamped commentary."""

    if max_join_gap_seconds < 0:
        raise ValueError("max_join_gap_seconds must be non-negative")
    if max_window_segments < 1:
        raise ValueError("max_window_segments must be positive")
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be between 0 and 1")
    normalized = normalize_transcript_segments(segments)
    candidates: list[TranscriptEvent] = []
    for text, start, end, indices in _windowed_segments(
        normalized,
        max_join_gap=max_join_gap_seconds,
        max_segments=max_window_segments,
    ):
        confidences = [normalized[index].confidence for index in indices]
        candidates.extend(_event_candidates(text, start, end, indices, confidences))
    deduplicated = deduplicate_events(candidates, max_gap_seconds=dedupe_window_seconds)
    return [event for event in deduplicated if event.confidence >= minimum_confidence]


def detect_events(
    segments: Iterable[TranscriptSegment | Mapping[str, Any]], **kwargs: Any
) -> list[TranscriptEvent]:
    """Backward-compatible short alias for :func:`detect_transcript_events`."""

    return detect_transcript_events(segments, **kwargs)
