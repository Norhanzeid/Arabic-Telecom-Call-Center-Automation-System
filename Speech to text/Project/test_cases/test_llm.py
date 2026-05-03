# -*- coding: utf-8 -*-
import os
import json
from dotenv import load_dotenv
from groq import Groq
from modules.speech_to_text.openai_whisper import transcribe_with_timestamps
from prompt_loader import load_prompt

# Load environment variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=api_key)

def diarize_with_llm(transcription):
    """Perform speaker diarization using Groq LLM."""
    transcription_json = json.dumps(transcription, ensure_ascii=False, indent=2)

    prompt = load_prompt("test_llm_diarization", transcription_json=transcription_json)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=4096
    )

    result_text = response.choices[0].message.content.strip()

    try:
        diarized = json.loads(result_text)
        return diarized
    except json.JSONDecodeError:
        print("Error parsing JSON from LLM response:")
        print(result_text)
        return None

if __name__ == "__main__":
    audio_file_path = "path/to/audio/file.wav"  # ضع مسار الصوت هنا

    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

    # Step 1: Whisper transcription
    transcription_result = transcribe_with_timestamps(audio_file_path, language="ar")
    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"]}
        for seg in transcription_result["segments"]
    ]
    print(f"Transcribed {len(segments)} segments from audio.")

    # Step 2: LLM diarization
    diarized_segments = diarize_with_llm(segments)
    if diarized_segments:
        # Save result to JSON
        output_file = "diarized_output.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(diarized_segments, f, ensure_ascii=False, indent=2)
        print(f"Diarization complete. Saved to {output_file}")
    else:
        print("Diarization failed.")