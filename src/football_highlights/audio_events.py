"""Local audio excitement detection for broadcast highlight candidates.

The detector is intentionally independent of speech services and play-by-play data.  It
extracts mono PCM audio with the locally installed ``ffmpeg`` executable, computes a
small set of robust envelope features with NumPy, and clusters unusually exciting
moments into timestamped candidates.
"""

from __future__ import annotations

import io
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


class AudioExtractionError(RuntimeError):
    """Raised when local ffmpeg extraction or WAV decoding fails."""


@dataclass(frozen=True)
class AudioEventConfig:
    """Deterministic settings for frame features and event clustering."""

    sample_rate: int = 16_000
    frame_seconds: float = 0.40
    hop_seconds: float = 0.10
    activation_threshold: float = 0.62
    minimum_event_seconds: float = 0.30
    merge_gap_seconds: float = 2.50
    pre_roll_seconds: float = 2.0
    post_roll_seconds: float = 3.0
    loudness_weight: float = 0.45
    onset_weight: float = 0.30
    peak_weight: float = 0.25

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.frame_seconds <= 0 or self.hop_seconds <= 0:
            raise ValueError("frame_seconds and hop_seconds must be positive")
        if self.hop_seconds > self.frame_seconds:
            raise ValueError("hop_seconds cannot exceed frame_seconds")
        if not 0 < self.activation_threshold < 1:
            raise ValueError("activation_threshold must be between zero and one")
        if self.minimum_event_seconds < 0 or self.merge_gap_seconds < 0:
            raise ValueError("event durations cannot be negative")
        if self.pre_roll_seconds < 0 or self.post_roll_seconds < 0:
            raise ValueError("roll durations cannot be negative")
        weights = (self.loudness_weight, self.onset_weight, self.peak_weight)
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("feature weights must be non-negative and non-zero")


@dataclass(frozen=True)
class AudioEvent:
    """One locally detected audio-excitement candidate.

    ``start`` and ``end`` include the configurable context roll and are ready to be
    passed to a clip boundary stage.  ``activity_start`` and ``activity_end`` retain
    the unpadded audio cluster for auditing and later refinement.
    """

    start: float
    end: float
    peak_time: float
    confidence: float
    activity_start: float
    activity_end: float
    loudness_score: float
    onset_score: float
    peak_score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class AudioFeatures:
    """Frame-level features used to produce audio events."""

    times: np.ndarray
    loudness_db: np.ndarray
    loudness_score: np.ndarray
    onset_score: np.ndarray
    peak_score: np.ndarray
    excitement_score: np.ndarray


def build_ffmpeg_audio_command(
    input_path: str | Path,
    *,
    sample_rate: int = 16_000,
    ffmpeg_bin: str = "ffmpeg",
) -> list[str]:
    """Build a local ffmpeg command that writes normalized mono WAV to stdout."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    return [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        "-f",
        "wav",
        "-",
    ]


def decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode PCM WAV bytes into finite mono float32 samples in ``[-1, 1]``."""

    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)
    except (wave.Error, EOFError) as exc:
        raise AudioExtractionError("ffmpeg returned invalid WAV audio") from exc

    if channels <= 0 or sample_rate <= 0 or sample_width not in {1, 2, 3, 4}:
        raise AudioExtractionError("WAV audio has unsupported metadata")
    if not raw:
        return np.empty(0, dtype=np.float32), sample_rate

    if sample_width == 1:
        values = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        values = (values - 128.0) / 128.0
    elif sample_width == 2:
        values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        values = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2_147_483_648.0
    else:
        bytes_array = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        unsigned = (
            bytes_array[:, 0].astype(np.int32)
            | (bytes_array[:, 1].astype(np.int32) << 8)
            | (bytes_array[:, 2].astype(np.int32) << 16)
        )
        signed = np.where(unsigned & 0x800000, unsigned - 0x1000000, unsigned)
        values = signed.astype(np.float32) / 8_388_608.0

    if values.size % channels:
        raise AudioExtractionError("WAV data is not aligned to its channel count")
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False), sample_rate


def extract_audio_ffmpeg(
    input_path: str | Path,
    *,
    sample_rate: int = 16_000,
    ffmpeg_bin: str = "ffmpeg",
    runner: Any = subprocess.run,
) -> tuple[np.ndarray, int]:
    """Extract local audio through ffmpeg and return mono samples plus sample rate."""

    command = build_ffmpeg_audio_command(input_path, sample_rate=sample_rate, ffmpeg_bin=ffmpeg_bin)
    try:
        completed = runner(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AudioExtractionError(f"ffmpeg audio extraction failed for {input_path}") from exc
    stdout = completed.stdout
    if isinstance(stdout, str):
        stdout = stdout.encode()
    if not isinstance(stdout, (bytes, bytearray)):
        raise AudioExtractionError("ffmpeg did not return WAV bytes on stdout")
    return decode_wav(bytes(stdout))


def _robust_score(values: np.ndarray) -> np.ndarray:
    """Map a feature to [0, 1] using median/MAD, resisting broadcast outliers."""

    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = max(1.4826 * mad, float(np.std(values)) * 0.25, 1e-6)
    z = (values - median) / scale
    return np.clip((z + 1.0) / 7.0, 0.0, 1.0).astype(np.float32)


def _frame_values(samples: np.ndarray, frame_size: int, hop_size: int) -> tuple[np.ndarray, ...]:
    """Calculate aligned RMS dB, onset proxy, and block-based peak values."""

    if samples.size == 0:
        empty = np.empty(0, dtype=np.float32)
        return empty, empty, empty
    frame_count = max(1, int(np.ceil(max(1, samples.size - frame_size + 1) / hop_size)))
    starts = np.arange(frame_count, dtype=np.int64) * hop_size
    ends = starts + frame_size
    padded = np.pad(samples, (0, max(0, int(ends[-1]) - samples.size)))
    squared_sum = np.concatenate(([0.0], np.cumsum(padded.astype(np.float64) ** 2)))
    rms = np.sqrt((squared_sum[ends] - squared_sum[starts]) / frame_size)
    loudness_db = (20.0 * np.log10(np.maximum(rms, 1e-7))).astype(np.float32)

    block_count = int(np.ceil(padded.size / hop_size))
    block_padded = np.pad(np.abs(samples), (0, max(0, block_count * hop_size - samples.size)))
    block_peaks = block_padded.reshape(block_count, hop_size).max(axis=1)
    block_window = max(1, int(np.ceil(frame_size / hop_size)))
    peak_windows = np.lib.stride_tricks.sliding_window_view(block_peaks, block_window)
    peak_values = peak_windows[:frame_count].max(axis=1)

    onset = np.maximum(np.diff(loudness_db, prepend=loudness_db[0]), 0.0)
    return loudness_db, onset.astype(np.float32), peak_values.astype(np.float32)


def compute_audio_features(
    samples: np.ndarray,
    sample_rate: int,
    *,
    config: AudioEventConfig | None = None,
) -> AudioFeatures:
    """Compute robust frame-level loudness, onset, peak, and combined scores."""

    settings = config or AudioEventConfig(sample_rate=sample_rate)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("samples must contain only finite values")
    frame_size = max(1, int(round(settings.frame_seconds * sample_rate)))
    hop_size = max(1, int(round(settings.hop_seconds * sample_rate)))
    loudness_db, onset, peak = _frame_values(values, frame_size, hop_size)
    loudness_score = _robust_score(loudness_db)
    onset_score = _robust_score(onset)
    peak_score = _robust_score(20.0 * np.log10(np.maximum(peak, 1e-7)))
    weight_total = settings.loudness_weight + settings.onset_weight + settings.peak_weight
    excitement = (
        settings.loudness_weight * loudness_score
        + settings.onset_weight * onset_score
        + settings.peak_weight * peak_score
    ) / weight_total
    times = (np.arange(loudness_db.size, dtype=np.float32) * settings.hop_seconds).astype(
        np.float32
    )
    return AudioFeatures(
        times=times,
        loudness_db=loudness_db,
        loudness_score=loudness_score,
        onset_score=onset_score,
        peak_score=peak_score,
        excitement_score=excitement.astype(np.float32),
    )


def _cluster_indices(
    times: np.ndarray,
    active: np.ndarray,
    *,
    frame_seconds: float,
    minimum_event_seconds: float,
    merge_gap_seconds: float,
) -> list[np.ndarray]:
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return []
    clusters: list[list[int]] = [[int(indices[0])]]
    for index in indices[1:]:
        gap = float(times[index] - times[clusters[-1][-1]] - frame_seconds)
        if gap <= merge_gap_seconds:
            clusters[-1].append(int(index))
        else:
            clusters.append([int(index)])
    minimum = max(0.0, minimum_event_seconds)
    return [
        np.asarray(cluster, dtype=np.int64)
        for cluster in clusters
        if float(times[cluster[-1]] - times[cluster[0]] + frame_seconds) >= minimum
    ]


def detect_audio_events(
    samples: np.ndarray,
    sample_rate: int,
    *,
    config: AudioEventConfig | None = None,
) -> list[AudioEvent]:
    """Detect and cluster unusual audio moments from a local PCM signal."""

    settings = config or AudioEventConfig(sample_rate=sample_rate)
    features = compute_audio_features(samples, sample_rate, config=settings)
    if features.times.size == 0:
        return []
    active = features.excitement_score >= settings.activation_threshold
    clusters = _cluster_indices(
        features.times,
        active,
        frame_seconds=settings.frame_seconds,
        minimum_event_seconds=settings.minimum_event_seconds,
        merge_gap_seconds=settings.merge_gap_seconds,
    )
    duration = len(np.asarray(samples).reshape(-1)) / sample_rate
    events: list[AudioEvent] = []
    for cluster in clusters:
        local_scores = features.excitement_score[cluster]
        peak_position = int(cluster[int(np.argmax(local_scores))])
        activity_start = float(features.times[cluster[0]])
        activity_end = min(duration, float(features.times[cluster[-1]] + settings.frame_seconds))
        loudness = float(np.max(features.loudness_score[cluster]))
        onset = float(np.max(features.onset_score[cluster]))
        peak = float(np.max(features.peak_score[cluster]))
        confidence = float(
            np.clip(
                0.55 * float(np.max(local_scores))
                + 0.20 * float(np.mean(local_scores))
                + 0.25 * min(1.0, (activity_end - activity_start) / 3.0),
                0.0,
                1.0,
            )
        )
        reasons = tuple(
            reason
            for reason, value in (
                ("crowd/commentary loudness", loudness),
                ("rapid audio onset", onset),
                ("short peak/transient", peak),
            )
            if value >= 0.65
        )
        events.append(
            AudioEvent(
                start=max(0.0, activity_start - settings.pre_roll_seconds),
                end=min(duration, activity_end + settings.post_roll_seconds),
                peak_time=float(features.times[peak_position] + settings.frame_seconds / 2),
                confidence=confidence,
                activity_start=activity_start,
                activity_end=activity_end,
                loudness_score=loudness,
                onset_score=onset,
                peak_score=peak,
                reasons=reasons,
            )
        )
    return events


def detect_audio_events_from_video(
    input_path: str | Path,
    *,
    config: AudioEventConfig | None = None,
    ffmpeg_bin: str = "ffmpeg",
    runner: Any = subprocess.run,
) -> list[AudioEvent]:
    """Extract a video's audio locally and return automatic excitement candidates."""

    settings = config or AudioEventConfig()
    samples, sample_rate = extract_audio_ffmpeg(
        input_path,
        sample_rate=settings.sample_rate,
        ffmpeg_bin=ffmpeg_bin,
        runner=runner,
    )
    return detect_audio_events(samples, sample_rate, config=settings)


__all__ = [
    "AudioEvent",
    "AudioEventConfig",
    "AudioExtractionError",
    "AudioFeatures",
    "build_ffmpeg_audio_command",
    "compute_audio_features",
    "decode_wav",
    "detect_audio_events",
    "detect_audio_events_from_video",
    "extract_audio_ffmpeg",
]
