"""
Speech-to-Text Module
Uses OpenAI Whisper API (cloud) for Arabic transcription.
No local model download required - uses Whisper large-v2 via API.
"""

import os
from .audio_preprocessor import preprocess_audio_simple
from .arabic_corrector import correct_arabic_text, correct_arabic_text_gpt, clean_number_dashes, apply_word_corrections
from .extract import extract_call_center_entities
from .openai_whisper import transcribe_with_openai


def _arabic_speech_to_text(audio, use_preprocessing=True):
    """
    Transcribes Arabic audio using OpenAI Whisper API (cloud).
    Uses Whisper large-v2 model without local download.
    
    Args:
        audio: Path to the audio file (supports .m4a, .mp3, .wav, .flac)
        use_preprocessing: Whether to apply audio preprocessing (default: True)
        
    Returns:
        str: Transcribed Arabic text
    """
    preprocessed_audio = None
    try:
        # Validate file exists
        if not os.path.exists(audio):
            raise Exception(f"Audio file not found: {audio}")

        print("🌐 Processing Arabic audio with OpenAI Whisper API...")
        print(f"📁 File: {audio}")

        audio_path = audio

        # Apply audio preprocessing if enabled
        if use_preprocessing:
            print("🔧 Applying audio preprocessing...")
            preprocessed_audio = preprocess_audio_simple(audio_path)
            if preprocessed_audio:
                audio_path = preprocessed_audio
                print("✅ Audio preprocessing completed")

        # Transcribe using OpenAI API (Whisper large-v2)
        print("🚀 Sending to OpenAI Whisper API (large-v2 model)...")
        text = transcribe_with_openai(audio_path, language="ar")

        print("✅ Arabic transcription completed!")
        return text.strip()

    except Exception as e:
        print(f"❌ Arabic transcription error: {e}")
        raise Exception(f"Arabic transcription error: {e}")
    finally:
        # Clean up temporary files
        if preprocessed_audio and os.path.exists(preprocessed_audio):
            try:
                os.unlink(preprocessed_audio)
            except:
                pass


def run(audio, config):
    """
    Main entry point for speech-to-text transcription.
    Uses OpenAI Whisper API for transcription (no local model).
    
    Args:
        audio: Path to the audio file
        config: Configuration dictionary with:
            - 'correct_arabic': True/False to enable Arabic text correction (default: True)
            - 'use_gpt_correction': True/False to use GPT for correction (default: False)
            - 'preprocess_audio': True/False to enable audio preprocessing (default: True)
            - 'extract_entities': True/False to enable entity extraction (default: False)
        
    Returns:
        str or dict: Transcribed text (corrected as configured), or dict with text and entities
    """
    correct_arabic = config.get("correct_arabic", True)
    use_gpt_correction = config.get("use_gpt_correction", False)
    use_preprocessing = config.get("preprocess_audio", True)
    extract_entities_flag = config.get("extract_entities", False)
    
    print(f"🌐 Language: Arabic")
    print(f"☁️  Using: OpenAI Whisper API (large-v2)")
    print(f"🔧 Audio preprocessing: {'enabled' if use_preprocessing else 'disabled'}")
    print(f"🤖 GPT correction: {'enabled' if use_gpt_correction else 'disabled (dictionary mode)'}")
    print(f"🔍 Entity extraction: {'enabled' if extract_entities_flag else 'disabled'}")
    
    # Get Whisper transcription via API
    text = _arabic_speech_to_text(audio, use_preprocessing)
    
    # Apply Arabic text correction if enabled
    if correct_arabic:
        if use_gpt_correction:
            print("🤖 Applying GPT-based Arabic text correction...")
            text = correct_arabic_text_gpt(text)
        else:
            print("🔄 Applying dictionary-based Arabic text correction...")
            text = correct_arabic_text(text)
    
    # Always apply word-level dictionary corrections (catches remaining errors)
    text = apply_word_corrections(text)
    
    # Always clean dashes from numbers (e.g. 0-551-234-567 -> 0551234567)
    text = clean_number_dashes(text)
    
    # Apply entity extraction if enabled
    if extract_entities_flag:
        print("📋 Extracting entities...")
        entities = extract_call_center_entities(text)
        return {
            'text': text,
            'entities': entities
        }
    
    return text


# =============================================================================
# Test / Demo
# =============================================================================
if __name__ == "__main__":
    # Example usage
    arabic_audio = r"C:\Users\Dell\Downloads\audio_ar.wav"
    
    config = {
        "correct_arabic": True,
        "preprocess_audio": True,
        "extract_entities": True
    }
    
    print(f"🎤 Testing speech-to-text with OpenAI API...")
    print(f"📁 Audio file: {arabic_audio}")
    
    try:
        result = run(arabic_audio, config)
        print(f"\n✅ Transcription Result:")
        print("-" * 40)
        if isinstance(result, dict):
            print("Text:", result['text'])
            print("\nEntities:")
            print(f"  Client Name: {result['entities']['client_name']}")
            print(f"  Phone Number: {result['entities']['phone_number']}")
            print(f"  Problem Statement: {result['entities']['problem_statement']}")
        else:
            print(result)
        print("-" * 40)
    except Exception as e:
        print(f"❌ Error: {e}")