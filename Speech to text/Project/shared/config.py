# Configuration file for Speech-to-Text app
# Now using OpenAI Whisper model locally via Hugging Face transformers

# Audio recording duration in seconds (used by record_audio function)
RECORDING_DURATION = 5

# =============================================================================
# Language Configuration
# =============================================================================
# Set LANGUAGE to "ar" for Arabic
LANGUAGE = "ar"  # Arabic only

# =============================================================================
# Whisper Model Configuration
# =============================================================================
# Arabic Model (via official Whisper library)
# Options: tiny(39MB), base(142MB), small(466MB), medium(1.5GB), large(2.9GB)
WHISPER_MODEL_ARABIC = "small"  # Best balance for Arabic with limited disk space

# Legacy alias for backward compatibility
WHISPER_MODEL = WHISPER_MODEL_ARABIC

# =============================================================================
# Audio Settings
# =============================================================================
SUPPORTED_FORMATS = [".m4a", ".mp3", ".wav", ".flac"]  # Supported audio formats

# =============================================================================
# Helper function to get config dictionary
# =============================================================================
def get_config():
    """
    Returns configuration dictionary for speech-to-text module.
    
    Usage:
        from config import get_config
        config = get_config()
        # Then pass to run function:
        # from Project.modules.speech_to_text.task import run
        # result = run(audio_path, config)
    """
    return {
        "language": LANGUAGE,
        "whisper_model_arabic": WHISPER_MODEL_ARABIC,
        "recording_duration": RECORDING_DURATION,
        "supported_formats": SUPPORTED_FORMATS
    }

# Note: This uses the actual OpenAI Whisper model running locally!
# No API keys required - completely offline after initial model download
# Much more accurate than traditional speech recognition engines