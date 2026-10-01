# Football Highlights

A local Python pipeline that discovers NFL highlights directly from a full-game
video and renders a 9:16 reel with the full broadcast over a heavily blurred
background. No play-by-play feed, event JSON,
manually selected timestamps, or cloud inference is used.

## Requirements

- macOS or Linux
- Python 3.12+
- `uv`
- FFmpeg and FFprobe

```bash
uv sync --extra analysis
```

Model assets may be downloaded once during setup. Inference, video, audio,
transcripts, OCR data, and rendered media stay on the machine.

## Automatic workflow

Analyze a recording using only the video:

```bash
uv run --extra analysis nfl-reel analyze-video /path/to/game.mkv \
  --output output/game/highlights.json
```

Analyze and render in one command:

```bash
uv run --extra analysis nfl-reel auto-reel /path/to/game.mkv \
  --output-root output/automatic-reel
```

The analysis is hierarchical:

1. Local audio analysis proposes exciting moments.
2. Local faster-whisper transcription identifies football event language.
3. RapidOCR reads the broadcast scorebug.
4. A strict live-play gate requires a wide pre-snap field view, coordinated
   snap motion, scorebug coverage, game-clock movement, and one continuous live
   shot. Commentary, studio footage, and replay-only candidates are rejected.
5. Accepted plays retain the full broadcast framing and are rendered at
   1080x1920 over a heavily blurred copy of the game, with original audio.

Intermediate manifests and transcripts are generated audit artifacts, not
manually authored inputs. Repeated runs reuse the cached transcript.

## Other commands

```bash
uv run nfl-reel inspect /path/to/game.mkv
uv run nfl-reel render /path/to/game.mkv output/game/highlights.json
uv run nfl-reel sample-scorebug /path/to/game.mkv -t 300 -o output/samples
```
