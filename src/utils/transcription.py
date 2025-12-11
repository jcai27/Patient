"""Audio transcription helpers using OpenAI Whisper."""
from pathlib import Path
from typing import Optional

import openai

from src.config import OPENAI_API_KEY


def transcribe_audio_file(
    audio_path: Path,
    model: str = "whisper-1",
    language: Optional[str] = None,
    temperature: float = 0.0,
) -> str:
    """
    Transcribe an audio file via OpenAI Whisper.

    Args:
        audio_path: Local path to the audio file.
        model: Whisper model name.
        language: Optional hint for the audio language.
        temperature: Sampling temperature (0.0 is deterministic).

    Returns:
        The transcription text with leading/trailing whitespace trimmed.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY must be set to transcribe audio.")

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    try:
        with open(audio_path, "rb") as audio_file:
            kwargs = {"model": model, "file": audio_file, "temperature": temperature}
            if language:
                kwargs["language"] = language
            response = client.audio.transcriptions.create(**kwargs)

        text = getattr(response, "text", None)
        if text is None:
            text = response.get("text") if isinstance(response, dict) else None
        return (text or "").strip()
    except Exception as exc:
        raise RuntimeError(f"Failed to transcribe '{audio_path.name}': {exc}") from exc
