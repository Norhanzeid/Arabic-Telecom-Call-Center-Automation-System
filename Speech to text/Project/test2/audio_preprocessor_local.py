"""
Local Audio Preprocessor Module
Simple audio conversion - no external dependencies beyond ffmpeg
"""

import os
import tempfile
import subprocess


def preprocess_audio_simple(audio_path: str, output_path: str = None) -> str:
    """
    Simple audio preprocessing (format conversion only).
    
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
            '-ar', '16000',      # 16kHz sample rate (optimal for Whisper)
            '-ac', '1',          # Mono channel
            '-acodec', 'pcm_s16le',
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        print("✅ Audio conversion completed!")
        return output_path
        
    except Exception as e:
        print(f"⚠️ Conversion failed: {e}")
        return audio_path


def preprocess_audio_full(audio_path: str, output_path: str = None) -> str:
    """
    Full audio preprocessing with noise reduction.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path for processed audio (optional)
        
    Returns:
        str: Path to processed audio file
    """
    if not os.path.exists(audio_path):
        raise Exception(f"Audio file not found: {audio_path}")
    
    print("🔧 Preprocessing audio...")
    
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
    
    try:
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-i', audio_path,
            '-af', (
                'highpass=f=200,'
                'lowpass=f=3000,'
                'afftdn=nf=-25,'
                'loudnorm=I=-16:LRA=11:TP=-1.5'
            ),
            '-ar', '16000',
            '-ac', '1',
            '-acodec', 'pcm_s16le',
            output_path
        ]
        
        subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
        print("✅ Audio preprocessing completed!")
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️ FFmpeg preprocessing failed, using simple conversion: {e}")
        return preprocess_audio_simple(audio_path, output_path)
    except FileNotFoundError:
        print("⚠️ FFmpeg not found, using original audio")
        return audio_path
