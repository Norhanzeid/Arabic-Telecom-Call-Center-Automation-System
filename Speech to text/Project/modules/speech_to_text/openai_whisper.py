"""
OpenAI Whisper API Client
Uses OpenAI's cloud-based Whisper for speech-to-text with optimal settings
"""
import os
import time
import logging
from openai import OpenAI

# Optional httpx import for richer exception typing (best-effort)
try:
    import httpx  # type: ignore
except Exception:
    httpx = None

# Lazy-initialize OpenAI client
_client = None

def get_client():
    """Get OpenAI client (lazy initialization)"""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        _client = OpenAI(api_key=api_key)
    return _client


def transcribe_with_openai(audio_file_path: str, language: str = None) -> str:
    """
    Transcribe audio using OpenAI's Whisper API (large-v2 model)
    
    Args:
        audio_file_path: Path to audio file
        language: Language code (None for auto-detect, 'ar' for Arabic)
    
    Returns:
        Transcribed text
    """
    client = get_client()
    max_retries = 3
    backoff_factor = 0.6

    for attempt in range(1, max_retries + 1):
        try:
            with open(audio_file_path, "rb") as audio_file:
                kwargs = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "text"
                }
                if language:
                    kwargs["language"] = language

                response = client.audio.transcriptions.create(**kwargs)
            return response
        except Exception as e:
            err_text = str(e)
            is_connection_error = False
            if httpx is not None and isinstance(e, httpx.RequestError):
                is_connection_error = True
            if isinstance(e, (ConnectionError, OSError)):
                is_connection_error = True
            if "connection" in err_text.lower() or "connection" in repr(e).lower():
                is_connection_error = True

            logging.exception("OpenAI Whisper call failed (attempt %s/%s)", attempt, max_retries)

            if not is_connection_error:
                raise Exception(f"OpenAI Whisper API error: {err_text}")

            if attempt == max_retries:
                raise Exception(f"OpenAI Whisper API connection error after {max_retries} attempts: {err_text}")

            sleep_for = backoff_factor * (2 ** (attempt - 1))
            time.sleep(sleep_for)


def transcribe_with_timestamps(audio_file_path: str, language: str = None) -> dict:
    """
    Transcribe audio with segments and timestamps for better diarization
    
    Args:
        audio_file_path: Path to audio file
        language: Language code (None for auto-detect, 'ar' for Arabic)
    
    Returns:
        Dict with text, segments, language, and duration
    """
    client = get_client()
    max_retries = 3
    backoff_factor = 0.6

    for attempt in range(1, max_retries + 1):
        try:
            with open(audio_file_path, "rb") as audio_file:
                kwargs = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "verbose_json",
                    "timestamp_granularities": ["segment"]
                }
                if language:
                    kwargs["language"] = language

                response = client.audio.transcriptions.create(**kwargs)

            # Convert TranscriptionSegment objects to plain dicts
            segments = []
            if hasattr(response, 'segments') and response.segments:
                for seg in response.segments:
                    segments.append({
                        "start": getattr(seg, 'start', 0.0),
                        "end": getattr(seg, 'end', 0.0),
                        "text": getattr(seg, 'text', '')
                    })

            return {
                "text": response.text,
                "segments": segments,
                "language": response.language if hasattr(response, 'language') else 'unknown',
                "duration": response.duration if hasattr(response, 'duration') else 0.0
            }
        except Exception as e:
            err_text = str(e)
            is_connection_error = False
            if httpx is not None and isinstance(e, httpx.RequestError):
                is_connection_error = True
            if isinstance(e, (ConnectionError, OSError)):
                is_connection_error = True
            if "connection" in err_text.lower() or "connection" in repr(e).lower():
                is_connection_error = True

            logging.exception("OpenAI Whisper call failed (attempt %s/%s)", attempt, max_retries)

            if not is_connection_error:
                raise Exception(f"OpenAI Whisper API error: {err_text}")

            if attempt == max_retries:
                raise Exception(f"OpenAI Whisper API connection error after {max_retries} attempts: {err_text}")

            sleep_for = backoff_factor * (2 ** (attempt - 1))
            time.sleep(sleep_for)


def validate_audio_format(filename: str) -> bool:
    """
    Validate if the audio file format is supported
    
    Args:
        filename: Name of the audio file
        
    Returns:
        True if supported format, False otherwise
    """
    supported_formats = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm', '.ogg', '.flac'}
    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in supported_formats
