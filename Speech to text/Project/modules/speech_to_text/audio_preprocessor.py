"""
Audio Preprocessing Module
Improves audio quality before speech-to-text conversion.
"""

import os
import tempfile
import subprocess


def preprocess_audio(audio_path: str, output_path: str = None) -> str:
    """
    Preprocesses audio file for better speech recognition.
    
    Steps:
    1. Convert to 16kHz mono WAV (optimal for Whisper)
    2. Normalize audio levels
    3. Apply noise reduction
    4. Remove silence
    
    Args:
        audio_path: Path to input audio file
        output_path: Path for processed audio (optional)
        
    Returns:
        str: Path to processed audio file
    """
    if not os.path.exists(audio_path):
        raise Exception(f"Audio file not found: {audio_path}")
    
    print("🔧 Preprocessing audio...")
    
    # Create output path if not provided
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    
    try:
        # FFmpeg command for audio preprocessing
        # -af: audio filters
        # highpass=f=200: remove low frequency noise
        # lowpass=f=3000: remove high frequency noise (keep speech range)
        # afftdn: noise reduction
        # silenceremove: remove silence at start/end
        # loudnorm: normalize loudness
        
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', audio_path,
            '-af', (
                'highpass=f=200,'           # Remove frequencies below 200Hz
                'lowpass=f=3000,'            # Remove frequencies above 3000Hz
                'afftdn=nf=-25,'             # Noise reduction
                'silenceremove=start_periods=1:start_silence=0.5:start_threshold=-50dB,'  # Remove silence at start
                'silenceremove=stop_periods=1:stop_silence=0.5:stop_threshold=-50dB,'     # Remove silence at end
                'loudnorm=I=-16:LRA=11:TP=-1.5'  # Normalize loudness
            ),
            '-ar', '16000',                  # Sample rate 16kHz (optimal for Whisper)
            '-ac', '1',                       # Mono channel
            '-acodec', 'pcm_s16le',          # 16-bit PCM
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        print("✅ Audio preprocessing completed!")
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️ FFmpeg preprocessing failed, using original audio: {e}")
        return audio_path
    except FileNotFoundError:
        print("⚠️ FFmpeg not found, using original audio")
        return audio_path


def preprocess_audio_simple(audio_path: str, output_path: str = None) -> str:
    """
    Simple audio preprocessing (just conversion).
    Use this if full preprocessing causes issues.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path for processed audio (optional)
        
    Returns:
        str: Path to processed audio file
    """
    if not os.path.exists(audio_path):
        raise Exception(f"Audio file not found: {audio_path}")
    
    print("🔧 Converting audio format...")
    
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', audio_path,
            '-ar', '16000',
            '-ac', '1',
            '-acodec', 'pcm_s16le',
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        print("✅ Audio conversion completed!")
        return output_path
        
    except Exception as e:
        print(f"⚠️ Conversion failed: {e}")
        return audio_path


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    print("🧪 Audio Preprocessing Module")
    print("Use preprocess_audio() for full preprocessing")
    print("Use preprocess_audio_simple() for basic conversion")
