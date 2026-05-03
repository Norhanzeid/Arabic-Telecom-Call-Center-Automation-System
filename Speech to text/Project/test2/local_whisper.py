"""
Local Whisper Module - Uses OpenAI Whisper locally (no API key required)
Downloads and runs Whisper model on your machine
"""
import os
import logging
import whisper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
_model = None
_model_name = None


def get_model(model_name: str = "large"):
    """
    Load Whisper model (cached for reuse)
    
    Args:
        model_name: Model size - tiny, base, small, medium, large, large-v2, large-v3
    
    Returns:
        Loaded Whisper model
    """
    global _model, _model_name
    
    if _model is None or _model_name != model_name:
        logger.info(f"🔄 Loading Whisper {model_name} model (this may take a while on first run)...")
        _model = whisper.load_model(model_name)
        _model_name = model_name
        logger.info(f"✅ Whisper {model_name} model loaded successfully!")
    
    return _model


def transcribe_local(audio_file_path: str, language: str = "ar", model_name: str = "medium") -> str:
    """
    Transcribe audio using local Whisper model
    
    Args:
        audio_file_path: Path to audio file
        language: Language code (ar for Arabic)
        model_name: Whisper model size
    
    Returns:
        Transcribed text
    """
    try:
        logger.info(f"🎤 Transcribing audio with local Whisper {model_name}...")
        
        model = get_model(model_name)
        
        # Transcribe with Arabic language
        result = model.transcribe(
            audio_file_path,
            language=language,
            task="transcribe",
            fp16=False,  # Use FP32 for CPU compatibility
            verbose=False
        )
        
        text = result["text"].strip()
        logger.info(f"✅ Transcription completed! Length: {len(text)} chars")
        
        return text
        
    except Exception as e:
        logger.error(f"❌ Local Whisper error: {str(e)}")
        raise Exception(f"Local Whisper transcription error: {str(e)}")


def transcribe_with_timestamps(audio_file_path: str, language: str = "ar", model_name: str = "medium") -> dict:
    """
    Transcribe audio with segment timestamps
    
    Returns:
        Dictionary with text and segments
    """
    try:
        model = get_model(model_name)
        
        result = model.transcribe(
            audio_file_path,
            language=language,
            task="transcribe",
            fp16=False,
            verbose=False
        )
        
        return {
            "text": result["text"].strip(),
            "segments": result.get("segments", []),
            "language": result.get("language", language)
        }
        
    except Exception as e:
        raise Exception(f"Local Whisper error: {str(e)}")


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("🧪 Testing Local Whisper Module")
    print("=" * 50)
    print("This module uses Whisper locally - no API key needed!")
    print(f"Model will be downloaded on first use (~2.9GB for large)")
    print("=" * 50)
