from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from football_highlights.media import probe_video
from football_highlights.models import HighlightManifest
from football_highlights.paths import ensure_input_video, job_directory
from football_highlights.reel import render_manifest
from football_highlights.scorebug import extract_scorebug_sample
from football_highlights.video_only_analysis import analyze_video_only

app = typer.Typer(no_args_is_help=True, help="Discover NFL highlights from a local video only.")


@app.command("auto-reel")
def auto_reel(
    source: Annotated[Path, typer.Argument(help="Full local path to the game video")],
    output_root: Annotated[Path, typer.Option("--output-root", "-o")] = Path(
        "output/automatic-reel"
    ),
    max_highlights: Annotated[int, typer.Option(help="Maximum highlights")] = 24,
    scoring_plays_only: Annotated[
        bool,
        typer.Option(
            "--scoring-plays-only/--all-highlights",
            help="Restrict discovery to touchdowns and made field goals",
        ),
    ] = False,
    whisper_model: Annotated[str, typer.Option(help="Local faster-whisper model")] = "tiny.en",
) -> None:
    """Discover and render a vertical reel using only the video."""
    video = ensure_input_video(source)
    destination = job_directory(output_root, video)
    manifest = analyze_video_only(
        video,
        max_highlights=max_highlights,
        scoring_plays_only=scoring_plays_only,
        whisper_model=whisper_model,
        transcript_cache=destination / "transcript.json",
        progress=typer.echo,
    )
    manifest.save(destination / "automatic-highlights.json")
    clips, reel = render_manifest(
        video,
        manifest,
        destination,
    )
    typer.echo(f"Automatically selected and rendered {len(clips)} highlights")
    typer.echo(str(reel.resolve()))


@app.command("analyze-video")
def analyze_video_command(
    source: Annotated[Path, typer.Argument(help="Full local path to the game video")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Generated manifest path")],
    max_highlights: Annotated[int, typer.Option(help="Maximum highlights")] = 24,
    scoring_plays_only: Annotated[
        bool,
        typer.Option(
            "--scoring-plays-only/--all-highlights",
            help="Restrict discovery to touchdowns and made field goals",
        ),
    ] = False,
    whisper_model: Annotated[str, typer.Option(help="Local faster-whisper model")] = "tiny.en",
) -> None:
    """Discover highlights from the video alone using local analysis."""
    video = ensure_input_video(source)
    manifest = analyze_video_only(
        video,
        max_highlights=max_highlights,
        scoring_plays_only=scoring_plays_only,
        whisper_model=whisper_model,
        transcript_cache=output.with_suffix(".transcript.json"),
        progress=typer.echo,
    )
    manifest.save(output)
    typer.echo(f"Generated {len(manifest.segments)} video-only highlights")
    typer.echo(str(output.resolve()))


@app.command()
def inspect(
    source: Annotated[Path, typer.Argument(help="Full local path to the game video")],
) -> None:
    """Inspect a source video without modifying it."""
    video = ensure_input_video(source)
    metadata = probe_video(video)
    typer.echo(
        json.dumps(
            {
                "source_path": str(video),
                "duration_seconds": metadata.duration,
                "width": metadata.width,
                "height": metadata.height,
                "fps": metadata.frame_rate,
                "has_audio": metadata.has_audio,
                "video_codec": metadata.video_codec,
            },
            indent=2,
        )
    )


@app.command()
def render(
    source: Annotated[Path, typer.Argument(help="Full local path to the game video")],
    manifest_path: Annotated[Path, typer.Argument(help="Generated highlight manifest")],
    output_root: Annotated[Path, typer.Option("--output-root", "-o")] = Path("output"),
) -> None:
    """Render a generated analysis manifest."""
    video = ensure_input_video(source)
    destination = job_directory(output_root, video)
    clips, reel = render_manifest(
        video,
        HighlightManifest.load(manifest_path),
        destination,
    )
    typer.echo(f"Rendered {len(clips)} clip(s)")
    typer.echo(str(reel.resolve()))


@app.command("sample-scorebug")
def sample_scorebug(
    source: Annotated[Path, typer.Argument(help="Full local path to the game video")],
    timestamps: Annotated[list[float], typer.Option("--timestamp", "-t")],
    output_directory: Annotated[Path, typer.Option("--output", "-o")] = Path(
        "output/scorebug-samples"
    ),
) -> None:
    """Extract scorebug crops for local OCR calibration."""
    video = ensure_input_video(source)
    metadata = probe_video(video)
    if not timestamps:
        raise typer.BadParameter("supply at least one --timestamp")
    output_directory.mkdir(parents=True, exist_ok=True)
    for timestamp in timestamps:
        destination = output_directory / f"scorebug-{timestamp:010.3f}.png"
        extract_scorebug_sample(video, destination, timestamp, metadata)
        typer.echo(str(destination.resolve()))


if __name__ == "__main__":
    app()
