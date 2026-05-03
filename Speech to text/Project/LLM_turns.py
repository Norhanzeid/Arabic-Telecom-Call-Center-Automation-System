# -*- coding: utf-8 -*-
import json
import logging
import os
import re
import time
from typing import Optional, Union

from groq import Groq
from dotenv import load_dotenv
from prompt_loader import load_prompt

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("LLM_RETRY_DELAY", "2.0"))

# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

client = Groq(api_key=api_key)


def extract_json_from_response(text: str) -> Optional[list]:
    """
    Extract JSON array from LLM response, handling markdown code blocks.

    Args:
        text: Raw LLM response text.

    Returns:
        Parsed JSON list, or None if parsing fails.
    """
    text = text.strip()

    # Try to extract from markdown code blocks (```json ... ``` or ``` ... ```)
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if json_match:
        text = json_match.group(1).strip()

    # Try to find a JSON array in the text
    if not text.startswith('['):
        array_match = re.search(r'(\[\s*\{[\s\S]*\}\s*\])', text)
        if array_match:
            text = array_match.group(1)

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        logger.warning("LLM response parsed but is not a list: %s", type(result))
        return None
    except json.JSONDecodeError as e:
        logger.error("Failed to parse LLM response as JSON: %s", e)
        return None


def validate_diarization_result(result: list, require_timestamps: bool = True) -> bool:
    """
    Validate that the diarization result has the expected structure.

    Args:
        result: Parsed JSON list from LLM response.
        require_timestamps: If True, require 'start' and 'end' keys (segment mode).
                           If False, only require 'speaker' and 'text' (raw text mode).

    Returns:
        True if valid, False otherwise.
    """
    required_keys = {"speaker", "text"}
    if require_timestamps:
        required_keys.update({"start", "end"})
    valid_speakers = {"Agent", "Customer", "Unknown"}

    for i, item in enumerate(result):
        if not isinstance(item, dict):
            logger.warning("Item %d is not a dict: %s", i, type(item))
            return False
        missing = required_keys - set(item.keys())
        if missing:
            logger.warning("Item %d missing keys: %s", i, missing)
            return False
        if item["speaker"] not in valid_speakers:
            logger.warning("Item %d has invalid speaker: '%s'", i, item["speaker"])
            return False
    return True

def diarize_with_llm(
    transcription: Union[str, list],
    max_retries: int = MAX_RETRIES,
    retry_delay: float = RETRY_DELAY,
) -> Optional[list]:
    """
    Perform speaker diarization on Arabic call center transcription using Groq LLM.

    Accepts EITHER:
      - A raw text string (plain transcription from Whisper)
      - A list of segment dicts with 'start', 'end', 'text' (verbose Whisper output)

    Args:
        transcription: Raw text string OR list of dicts with 'start', 'end', 'text'.
        max_retries: Number of retry attempts on failure.
        retry_delay: Seconds to wait between retries.

    Returns:
        List of dicts with 'speaker', 'start', 'end', 'text' (when segments provided),
        or list of dicts with 'speaker', 'text' (when raw text provided),
        or None on failure.
    """
    # Determine input type and build the transcription content for the prompt
    is_raw_text = isinstance(transcription, str)

    if is_raw_text:
        transcription_content = transcription.strip()
    else:
        transcription_content = json.dumps(transcription, ensure_ascii=False, indent=2)

    # Load the system prompt from external file
    system_prompt = load_prompt("llm_turns_system")

    # Load the user prompt template based on input type
    if is_raw_text:
        user_prompt = load_prompt("llm_turns_user_raw", transcription_content=transcription_content)
    else:
        user_prompt = load_prompt("llm_turns_user_segments", transcription_content=transcription_content)

    for attempt in range(1, max_retries + 1):
        try:
            logger.info("API call attempt %d/%d (model: %s)", attempt, max_retries, GROQ_MODEL)

            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,  # Low temperature for consistent output
                max_tokens=4096,
            )

            result_text = response.choices[0].message.content
            if not result_text:
                logger.warning("Empty response from LLM on attempt %d", attempt)
                continue

            diarized = extract_json_from_response(result_text)
            if diarized is None:
                logger.warning("Failed to parse JSON on attempt %d", attempt)
                if attempt < max_retries:
                    time.sleep(retry_delay)
                continue

            if not validate_diarization_result(diarized, require_timestamps=not is_raw_text):
                logger.warning("Invalid diarization structure on attempt %d", attempt)
                if attempt < max_retries:
                    time.sleep(retry_delay)
                continue

            logger.info("Diarization successful: %d segments identified", len(diarized))
            return diarized

        except Exception as e:
            logger.error("API call failed on attempt %d/%d: %s", attempt, max_retries, e)
            if attempt < max_retries:
                time.sleep(retry_delay)

    logger.error("All %d attempts failed for diarization", max_retries)
    return None

# Example usage
if __name__ == "__main__":
    # Enable logging for demo
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    # Pre-segmented sample transcript (simulating Whisper output)
    segments = [
        {"start": 0.0, "end": 8.0,
         "text": "مساء الخير يا فندم شكرا لاتصالك ب WE للاتصالات احمد في خدمتك ممكن اتشرف باسم حضرتك؟"},
        {"start": 8.0, "end": 20.0,
         "text": "اهلا استاذ احمد انا اسمي هبة والحقيقة ان انا موجودة علي خدمة الانتظار بقالي فترة طويلة جدا"},
        {"start": 20.0, "end": 32.0,
         "text": "انا بعتذر جدا يا فندم الوقت ده من اليوم بيكون فيه ضغط ومكالمات كبير جدا اسفين جدا علي وقت حضرتك ممكن حضرتك تبلغيني بالرقم المتاح عليه الخدمة؟"},
        {"start": 32.0, "end": 36.0,
         "text": "طبعا الرقم هو 01274714823"},
        {"start": 36.0, "end": 41.0,
         "text": "تمام يا فندم هل الخط مسجل باسم حضرتك؟"},
        {"start": 41.0, "end": 43.0,
         "text": "مسجل باسمي"},
        {"start": 43.0, "end": 48.0,
         "text": "طيب ممكن توضحي لي الاسم بالكامل يا فندم؟"},
        {"start": 48.0, "end": 51.0,
         "text": "هبة عاطف الناغي عفيفي"},
        {"start": 51.0, "end": 58.0,
         "text": "تمام شكرا جدا يا استاذة هبة اقدر اساعدك في ايه يا فندم؟"},
        {"start": 58.0, "end": 120.0,
         "text": "الحقيقة يا استاذ احمد انا عايز اقول لك ان انا بالضبط علي بعد 30 دقيقة من اني اتفصل من شغلي انا كنت بشتغل والانترنت شغال مشكلة طبيعي وبعد ما حضرت ملفات الاكسل والاميلات اللي كنت محتاج ابعتها قبل ما اضبط علي الارسال فجاة الانترنت فصل وعلامة الانترنت علي جهاز الكمبيوتر بتقول ان مفيش انترنت والموضوع مش منطقي خالص لاني دفعت الفاتورة في معادها الساعة دلوقتي 9 وانا محتاج ابعت الشغل بتاعي الساعة 10 من الكتير فمن فضلك استاذ احمد انا محتاج حالة لان الوقت لو عد من غير ما ابعت الاميلات دي هتكون مشكلة كبيرة في شغلي."},
    ]

    result = diarize_with_llm(segments)
    if result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("Failed to diarize transcription")
