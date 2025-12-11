#!/usr/bin/env python3
"""Script to ingest a transcript or audio file via Whisper."""

import argparse
import sys
from pathlib import Path

from src.ingest.transcript import TranscriptIngester
from src.utils.transcription import transcribe_audio_file


AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".aac"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe (if needed) and ingest persona data.")
    parser.add_argument(
        "source_path",
        nargs="?",
        default="transcript_cleaned.txt",
        help="Path to the transcript text or audio file (mp3/wav).",
    )
    parser.add_argument(
        "--persona-name",
        default="VirtualHuman",
        help="Persona directory name to create under the persona store.",
    )
    parser.add_argument(
        "--audio-dir",
        default=None,
        help="Optional directory containing wav files for stress analysis.",
    )
    return parser.parse_args()


def _load_transcript(source_path: Path) -> str:
    if not source_path.exists():
        raise FileNotFoundError(f"Source path not found: {source_path}")

    if source_path.suffix.lower() in AUDIO_EXTENSIONS:
        print(f"🧠 Transcribing audio file with Whisper: {source_path.name}")
        return transcribe_audio_file(source_path)

    print(f"📖 Reading transcript from: {source_path}")
    return source_path.read_text(encoding="utf-8")


def main():
    args = _parse_args()
    persona_name = args.persona_name
    source_path = Path(args.source_path)

    try:
        transcript = _load_transcript(source_path)
    except Exception as exc:
        print(f"❌ Failed to load transcript: {exc}")
        sys.exit(1)

    print(f"   Transcript length: {len(transcript)} characters")
    print(f"   Number of words: {len(transcript.split())}")
    print()

    print(f"🤖 Starting ingestion for persona: {persona_name}")
    print("   This will make multiple LLM calls and may take a few minutes...")
    print()

    try:
        ingester = TranscriptIngester()

        print("   Step 1: Chunking transcript...")
        result = ingester.ingest(
            transcript_path=str(source_path),
            persona_name=persona_name,
            transcript_text=transcript,
            audio_dir=args.audio_dir,
        )

        print()
        print("✅ Ingestion complete!")
        print(f"   Persona: {result['persona_name']}")
        print(f"   Facts extracted: {result['facts_count']}")
        print(f"   Examples generated: {result['examples_count']}")
        print(f"   Chunks created: {result.get('chunks_count', 'N/A')}")
        print(f"   Status: {result['status']}")
        print()
        print(f"📁 Persona artifacts saved to: persona/{persona_name}/")

    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

