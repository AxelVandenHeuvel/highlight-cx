"""Local replay, studio, and commercial rejection for broadcast clips.

The detector is deliberately independent of play-by-play and transcription.
It consumes a short sequence of visual signals and decides whether the
sequence looks like original live game action or a replay/studio/commercial
segment.  The pure functions are deterministic and easy to test; the OpenCV
sampler is only a convenience for producing those signals from a local file.

This is a rejection gate, not a highlight scorer.  Audio and event detectors
may propose a candidate, but this module can remove candidates whose visual
sequence is dominated by replay graphics, wipes, studio faces, or commercial
shots.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # Keep the pure decision API importable without OpenCV.
    import cv2
except ImportError:  # pragma: no cover - only relevant to minimal installs
    cv2 = None  # type: ignore[assignment]


def _bounded(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _validate_score(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")


@dataclass(frozen=True)
class FrameSignals:
    """Deterministic visual signals for one sampled frame.

    Values are normalized to ``0..1``.  ``scorebug_visible`` may be ``None``
    when a caller cannot measure it; unknown samples are excluded from the
    scorebug-absence ratio instead of being treated as absent.
    """

    timestamp: float
    scorebug_visible: bool | None = None
    replay_logo_score: float = 0.0
    wipe_score: float = 0.0
    motion_score: float = 0.0
    frame_delta_score: float = 0.0
    face_ratio: float = 0.0
    green_field_ratio: float = 0.0
    abrupt_cut_score: float = 0.0
    frame_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        for name in (
            "replay_logo_score",
            "wipe_score",
            "motion_score",
            "frame_delta_score",
            "face_ratio",
            "green_field_ratio",
            "abrupt_cut_score",
        ):
            _validate_score(name, getattr(self, name))


# More explicit alias for callers that use the term observation.
FrameObservation = FrameSignals


@dataclass(frozen=True)
class ReplayDetectionConfig:
    """Thresholds and evidence weights for :func:`analyze_sequence`."""

    scorebug_absence_threshold: float = 0.60
    replay_logo_threshold: float = 0.60
    wipe_threshold: float = 0.60
    slow_motion_threshold: float = 0.62
    repeated_fingerprint_threshold: float = 0.20
    studio_frame_threshold: float = 0.35
    face_ratio_threshold: float = 0.18
    studio_green_max: float = 0.32
    abrupt_cut_threshold: float = 0.65
    rejection_threshold: float = 0.48
    minimum_frames: int = 2

    def __post_init__(self) -> None:
        for name in (
            "scorebug_absence_threshold",
            "replay_logo_threshold",
            "wipe_threshold",
            "slow_motion_threshold",
            "repeated_fingerprint_threshold",
            "studio_frame_threshold",
            "face_ratio_threshold",
            "studio_green_max",
            "abrupt_cut_threshold",
            "rejection_threshold",
        ):
            _validate_score(name, getattr(self, name))
        if self.minimum_frames < 1:
            raise ValueError("minimum_frames must be positive")


@dataclass(frozen=True)
class RejectionDecision:
    """The explainable result of the replay/studio/commercial gate."""

    rejected: bool
    confidence: float
    reasons: tuple[str, ...]
    evidence: tuple[tuple[str, float], ...] = ()
    start_timestamp: float | None = None
    end_timestamp: float | None = None

    def __post_init__(self) -> None:
        _validate_score("confidence", self.confidence)
        if self.start_timestamp is not None and self.start_timestamp < 0:
            raise ValueError("start_timestamp must be non-negative")
        if self.end_timestamp is not None and self.end_timestamp < 0:
            raise ValueError("end_timestamp must be non-negative")
        if (
            self.start_timestamp is not None
            and self.end_timestamp is not None
            and self.end_timestamp < self.start_timestamp
        ):
            raise ValueError("end_timestamp must not precede start_timestamp")


@dataclass(frozen=True)
class VideoSample:
    """Signals and basic metadata returned by :func:`sample_video`."""

    frames: tuple[FrameSignals, ...]
    fps: float
    width: int
    height: int
    duration: float


ScorebugDetector = Callable[[np.ndarray, float], bool | None]


def scorebug_absence_ratio(
    frames: Iterable[FrameSignals],
) -> float | None:
    """Return the fraction of known scorebug readings that are absent."""

    known = [frame.scorebug_visible for frame in frames if frame.scorebug_visible is not None]
    if not known:
        return None
    return sum(not visible for visible in known) / len(known)


def _threshold_ratio(
    frames: Sequence[FrameSignals],
    value: Callable[[FrameSignals], float],
    threshold: float,
) -> float:
    if not frames:
        return 0.0
    return sum(value(frame) >= threshold for frame in frames) / len(frames)


def repeated_fingerprint_ratio(frames: Iterable[FrameSignals]) -> float:
    """Measure non-adjacent repeated frames using supplied fingerprints.

    Adjacent duplicates are normal in a low-motion broadcast and are not
    counted.  A repeated fingerprint later in the sequence is stronger
    evidence of a replay loop, frozen graphic, or commercial bumper.
    """

    fingerprints = [frame.frame_fingerprint for frame in frames]
    known = [fingerprint for fingerprint in fingerprints if fingerprint]
    if not known:
        return 0.0
    last_index: dict[str, int] = {}
    repeated = 0
    known_index = 0
    for fingerprint in fingerprints:
        if not fingerprint:
            continue
        previous = last_index.get(fingerprint)
        if previous is not None and known_index - previous > 1:
            repeated += 1
        last_index[fingerprint] = known_index
        known_index += 1
    return repeated / len(known)


def slow_motion_cadence_proxy(
    frames: Iterable[FrameSignals],
) -> float:
    """Estimate slow motion from low cadence and near-duplicate frames.

    This is intentionally a cadence proxy rather than optical-flow semantics:
    slow motion commonly produces low frame-to-frame change and a run of very
    similar frames.  It is combined with replay evidence before rejection.
    """

    ordered = sorted(frames, key=lambda frame: frame.timestamp)
    if not ordered:
        return 0.0
    low_motion = sum(frame.motion_score <= 0.34 for frame in ordered) / len(ordered)
    low_delta = sum(frame.frame_delta_score <= 0.18 for frame in ordered) / len(ordered)
    cadence = 0.55 * low_motion + 0.45 * low_delta
    return _bounded(cadence)


def studio_low_green_ratio(
    frames: Iterable[FrameSignals],
    *,
    config: ReplayDetectionConfig | None = None,
) -> float:
    """Return the fraction of frames resembling a face-led studio shot."""

    active = config or ReplayDetectionConfig()
    ordered = tuple(frames)
    return _threshold_ratio(
        ordered,
        lambda frame: float(
            frame.face_ratio >= active.face_ratio_threshold
            and frame.green_field_ratio <= active.studio_green_max
        ),
        0.5,
    )


def analyze_sequence(
    frames: Iterable[FrameSignals],
    *,
    config: ReplayDetectionConfig | None = None,
) -> RejectionDecision:
    """Classify a visual sequence as live-looking or non-live.

    The result is deterministic for a given sequence.  Confidence is a
    weighted evidence score with small bonuses for combinations that are much
    more diagnostic together, such as a replay wipe plus a replay logo or a
    studio face shot plus scorebug absence.
    """

    active = config or ReplayDetectionConfig()
    ordered = tuple(sorted(frames, key=lambda frame: frame.timestamp))
    if not ordered:
        return RejectionDecision(False, 0.0, ("no visual samples",), ())

    absence = scorebug_absence_ratio(ordered)
    absence_score = absence if absence is not None else 0.0
    logo_score = _threshold_ratio(
        ordered, lambda frame: frame.replay_logo_score, active.replay_logo_threshold
    )
    wipe_score = _threshold_ratio(ordered, lambda frame: frame.wipe_score, active.wipe_threshold)
    slow_score = slow_motion_cadence_proxy(ordered)
    repeat_score = repeated_fingerprint_ratio(ordered)
    studio_score = _threshold_ratio(
        ordered,
        lambda frame: float(
            frame.face_ratio >= active.face_ratio_threshold
            and frame.green_field_ratio <= active.studio_green_max
        ),
        0.5,
    )
    cut_score = _threshold_ratio(
        ordered, lambda frame: frame.abrupt_cut_score, active.abrupt_cut_threshold
    )

    evidence_values = {
        "scorebug_absence": absence_score,
        "replay_logo": logo_score,
        "replay_wipe": wipe_score,
        "slow_motion_cadence": slow_score,
        "repeated_frames": repeat_score,
        "studio_face_low_green": studio_score,
        "abrupt_cuts": cut_score,
    }
    weights = {
        "scorebug_absence": 0.22,
        "replay_logo": 0.18,
        "replay_wipe": 0.18,
        "slow_motion_cadence": 0.10,
        "repeated_frames": 0.10,
        "studio_face_low_green": 0.17,
        "abrupt_cuts": 0.05,
    }
    confidence = sum(weights[name] * value for name, value in evidence_values.items())
    confidence += 0.24 * min(logo_score, wipe_score)
    confidence += 0.18 * min(studio_score, absence_score)
    confidence += 0.10 * min(repeat_score, slow_score)
    confidence = _bounded(confidence)

    reasons: list[str] = []
    if absence is not None and absence >= active.scorebug_absence_threshold:
        reasons.append(f"scorebug absent in {absence:.0%} of sampled frames")
    if logo_score >= active.studio_frame_threshold:
        reasons.append(f"replay/logo graphics persist across {logo_score:.0%} of frames")
    if wipe_score >= active.studio_frame_threshold:
        reasons.append(f"broadcast wipe transitions detected in {wipe_score:.0%} of frames")
    if slow_score >= active.slow_motion_threshold:
        reasons.append(f"slow-motion cadence proxy is high ({slow_score:.2f})")
    if repeat_score >= active.repeated_fingerprint_threshold:
        reasons.append(f"non-adjacent frame fingerprints repeat ({repeat_score:.0%})")
    if studio_score >= active.studio_frame_threshold:
        reasons.append(f"face-led low-green frames detected ({studio_score:.0%})")
    if cut_score >= active.studio_frame_threshold:
        reasons.append(f"abrupt cuts detected in {cut_score:.0%} of frames")
    if logo_score >= active.studio_frame_threshold and wipe_score >= active.studio_frame_threshold:
        reasons.append("replay logo and wipe evidence agree")
    if (
        studio_score >= active.studio_frame_threshold
        and absence_score >= active.scorebug_absence_threshold
    ):
        reasons.append("studio-like frames coincide with scorebug absence")
    if not reasons:
        reasons.append("visual sequence remains consistent with live field action")

    start = ordered[0].timestamp
    end = ordered[-1].timestamp
    evidence = tuple((name, round(value, 6)) for name, value in evidence_values.items())
    return RejectionDecision(
        rejected=len(ordered) >= active.minimum_frames and confidence >= active.rejection_threshold,
        confidence=confidence,
        reasons=tuple(reasons),
        evidence=evidence,
        start_timestamp=start,
        end_timestamp=end,
    )


def reject_non_live_sequence(
    frames: Iterable[FrameSignals],
    *,
    config: ReplayDetectionConfig | None = None,
) -> RejectionDecision:
    """Readable alias for using the module as a candidate rejection gate."""

    return analyze_sequence(frames, config=config)


def is_replay_or_non_live(
    frames: Iterable[FrameSignals],
    *,
    config: ReplayDetectionConfig | None = None,
) -> bool:
    """Return only the rejection decision for pipeline filters."""

    return analyze_sequence(frames, config=config).rejected


def _crop(frame: np.ndarray, region: tuple[float, float, float, float]) -> np.ndarray:
    height, width = frame.shape[:2]
    x, y, crop_width, crop_height = region
    left = max(0, min(width - 1, round(x * width)))
    top = max(0, min(height - 1, round(y * height)))
    right = max(left + 1, min(width, round((x + crop_width) * width)))
    bottom = max(top + 1, min(height, round((y + crop_height) * height)))
    return frame[top:bottom, left:right]


def _graphic_score(crop: np.ndarray) -> float:
    """Estimate a persistent on-screen graphic from edges and saturation."""

    if cv2 is None or crop.size == 0:  # pragma: no cover
        return 0.0
    small = cv2.resize(crop, (96, 54), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 180)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size
    graphic_pixels = cv2.inRange(hsv, (0, 45, 100), (179, 255, 255))
    graphic_ratio = float(np.count_nonzero(graphic_pixels)) / graphic_pixels.size
    return _bounded(4.0 * edge_ratio + 1.6 * graphic_ratio)


def _green_ratio(frame: np.ndarray) -> float:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (30, 35, 25), (95, 255, 255))
    return float(np.count_nonzero(mask)) / mask.size


def _face_ratio(frame: np.ndarray, detector: Any) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    area = sum(width * height for x, y, width, height in faces)
    return _bounded(area / float(frame.shape[0] * frame.shape[1]))


def _fingerprint(frame: np.ndarray) -> str:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tiny = cv2.resize(gray, (16, 9), interpolation=cv2.INTER_AREA)
    normalized = (tiny >= float(np.mean(tiny))).astype(np.uint8)
    return hashlib.blake2b(normalized.tobytes(), digest_size=8).hexdigest()


def _wipe_score(previous: np.ndarray | None, current: np.ndarray) -> float:
    if previous is None:
        return 0.0
    difference = cv2.absdiff(previous, current)
    changed = (difference > 35).astype(np.uint8)
    changed_ratio = float(np.mean(changed))
    row_peak = float(np.max(np.mean(changed, axis=1)))
    column_peak = float(np.max(np.mean(changed, axis=0)))
    band_score = max(row_peak, column_peak)
    return _bounded((changed_ratio - 0.10) * 3.0 + (band_score - 0.25) * 1.4)


def _default_scorebug_visible(crop: np.ndarray) -> bool:
    return _graphic_score(crop) >= 0.22


def sample_video(
    source: str | Path,
    *,
    start: float = 0.0,
    end: float | None = None,
    sample_fps: float = 4.0,
    scorebug_detector: ScorebugDetector | None = None,
    scorebug_region: tuple[float, float, float, float] = (0.24, 0.76, 0.52, 0.22),
    replay_logo_regions: Sequence[tuple[float, float, float, float]] = (
        (0.0, 0.0, 0.24, 0.22),
        (0.76, 0.0, 0.24, 0.22),
        (0.0, 0.76, 0.28, 0.24),
        (0.72, 0.76, 0.28, 0.24),
    ),
) -> VideoSample:
    """Sample a local video into replay-gate signals using OpenCV only.

    The default detectors are deliberately generic.  A local OCR/graphics
    detector can be injected for ``scorebug_detector`` without changing the
    deterministic decision layer.  No network, cloud model, or play-by-play
    input is used.
    """

    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if start < 0 or (end is not None and end <= start):
        raise ValueError("sample window must have nonnegative start and end after start")
    if cv2 is None:  # pragma: no cover
        raise RuntimeError("OpenCV is required for video sampling")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise OSError(f"could not open video: {source}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        duration = count / fps if count else 0.0
        step = max(1, round(fps / sample_fps))
        face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        previous_gray: np.ndarray | None = None
        previous_small: np.ndarray | None = None
        observations: list[FrameSignals] = []
        capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
        frame_index = max(0, round(start * fps))
        first_frame = frame_index
        while True:
            success, frame = capture.read()
            if not success:
                break
            timestamp = frame_index / fps
            if end is not None and timestamp > end:
                break
            if (frame_index - first_frame) % step:
                frame_index += 1
                continue
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if previous_gray is None:
                delta = 0.0
                motion = 0.0
            else:
                raw_delta = float(np.mean(cv2.absdiff(gray, previous_gray))) / 255.0
                delta = _bounded(raw_delta * 4.0)
                motion = delta
            scorebug_crop = _crop(frame, scorebug_region)
            scorebug = (
                scorebug_detector(scorebug_crop, timestamp)
                if scorebug_detector is not None
                else _default_scorebug_visible(scorebug_crop)
            )
            logo_scores = [_graphic_score(_crop(frame, region)) for region in replay_logo_regions]
            logo_score = _bounded(max(logo_scores, default=0.0))
            observations.append(
                FrameSignals(
                    timestamp=timestamp,
                    scorebug_visible=scorebug,
                    replay_logo_score=logo_score,
                    wipe_score=_wipe_score(previous_small, small),
                    motion_score=motion,
                    frame_delta_score=delta,
                    face_ratio=_face_ratio(small, face_detector),
                    green_field_ratio=_green_ratio(small),
                    abrupt_cut_score=_bounded(delta * 1.8),
                    frame_fingerprint=_fingerprint(small),
                )
            )
            previous_gray = gray
            previous_small = small
            frame_index += 1
        if observations:
            duration = max(duration, observations[-1].timestamp)
        return VideoSample(tuple(observations), fps, width, height, duration)
    finally:
        capture.release()


def analyze_video(
    source: str | Path,
    *,
    sample_fps: float = 4.0,
    scorebug_detector: ScorebugDetector | None = None,
    config: ReplayDetectionConfig | None = None,
) -> RejectionDecision:
    """Sample a local video and return its replay/non-live decision."""

    sample = sample_video(
        source,
        sample_fps=sample_fps,
        scorebug_detector=scorebug_detector,
    )
    return analyze_sequence(sample.frames, config=config)


__all__ = [
    "FrameObservation",
    "FrameSignals",
    "RejectionDecision",
    "ReplayDetectionConfig",
    "VideoSample",
    "analyze_sequence",
    "analyze_video",
    "is_replay_or_non_live",
    "reject_non_live_sequence",
    "repeated_fingerprint_ratio",
    "sample_video",
    "scorebug_absence_ratio",
    "slow_motion_cadence_proxy",
    "studio_low_green_ratio",
]
