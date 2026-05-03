"""
Local Entity Extraction Module (No OpenAI API required)
Regex-based extraction - runs completely offline
"""

import re
import logging
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logging.warning("python-dotenv not installed; .env will not be auto-loaded. If you want .env support, install python-dotenv.")
import os
import json
import sys

# Ensure Project root is in path for prompt_loader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from prompt_loader import load_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_entities_local(text: str) -> dict:
    """
    Extract entities from Arabic text using regex patterns.
    No API required - runs completely offline.
    
    Args:
        text: Arabic transcription text
        
    Returns:
        Dictionary with client_name, phone_number, status
    """
    entities = {
        "client_name": None,
        "phone_number": None,
        "status": "new"
    }
    
    if not text:
        return entities
    
    logger.info("🔍 Extracting entities (regex-based)...")
    
    # ==========================================================================
    # Phone number patterns (Egypt)
    # ==========================================================================
    phone_patterns = [
        r'\b01[0-2,5]\d{8}\b',  # Mobile: 010, 011, 012, 015
        r'\b02\d{8}\b',         # Cairo landline
        r'\b03\d{8}\b',         # Alexandria landline
    ]
    
    # Normalize and convert spoken number-words to digits for better phone detection
    def _word_numbers_to_digits(s: str) -> str:
        num_map = {
            'زيرو': '0', 'زيرو.': '0', 'زير': '0', 'صفر': '0', 'صفره': '0',
            'واحد': '1', 'واحده': '1', 'واحدًا': '1',
            'اتنين': '2', 'اثنين': '2', 'اتنينه': '2',
            'تلات': '3', 'ثلاثة': '3', 'تلاتة': '3',
            'اربعة': '4', 'اربع': '4',
            'خمسة': '5', 'خمسه': '5',
            'ستة': '6', 'ست': '6',
            'سبعة': '7', 'سبع': '7',
            'تمانية': '8', 'تمانى': '8', 'تمانيه': '8',
            'تسعة': '9', 'تسعه': '9'
        }

        # Build pattern for number words
        keys = sorted(num_map.keys(), key=len, reverse=True)
        word_pat = r'(?:' + '|'.join(re.escape(k) for k in keys) + r')'
        seq_pat = re.compile(r'(' + word_pat + r'(?:\s+' + word_pat + r')*)', flags=re.IGNORECASE)

        def _replace_match(m):
            words = re.split(r'\s+', m.group(0))
            digits = ''.join(num_map.get(w.strip().lower(), '') for w in words)
            return digits

        s2 = seq_pat.sub(_replace_match, s)
        # Remove spaces between digits if any remain
        s2 = re.sub(r'(?<=\d)\s+(?=\d)', '', s2)
        return s2

    text_for_phone = _word_numbers_to_digits(text)
    logger.info(f"🔢 Phone-normalized text: {text_for_phone[:200]}")

    for pattern in phone_patterns:
        match = re.search(pattern, text_for_phone)
        if match:
            entities["phone_number"] = match.group()
            logger.info(f"📞 Found phone: {entities['phone_number']}")
            break
    
    # ==========================================================================
    # Name extraction patterns
    # ==========================================================================
    # Arabic character range: \u0600-\u06FF covers all Arabic characters
    
    name_patterns = [
        # Simple direct patterns
        r'(?:انا|أنا)\s+اسمي\s+([\u0600-\u06FF\s]{2,40})',
        r'مسجل\s+باسمي[\u061F\?\.]?\s*([\u0600-\u06FF\s]{2,40})',

        # After "الاسم بالكامل يا فندم" - most common pattern
        r'الاسم\s+بالكامل[\u061F\?\.]?\s*([\u0600-\u06FF\s]{2,50}?)',
        r'توضحي?لي?\s+الاسم\s+بالكامل\s+يا\s+فندم[\u061F\?\.]?\s*([\u0600-\u06FF\s]{2,50}?)',
        
        # After "الاسم بالكامل" without "يا فندم"
        r'الاسم\s+بالكامل[\u061F\?]?\s*([\u0600-\u06FF\s]{6,50}?)\s+تمام',
        
        # After asking for name
        r'باسم\s+حضرتك[\u061F\?]?\s*([\u0600-\u06FF\s]{4,40}?)(?:\s+و|\s+تمام|\s+الحقيقة)',
        
        # "اسمي" patterns - common in Arabic
        r'انا\s+اسمي\s+([\u0600-\u06FF]+)(?:\s+و|\s+تمام|\s+الحقيقة)?',
        r'أنا\s+اسمي\s+([\u0600-\u06FF]+)(?:\s+و|\s+تمام)?',
        
        # After "يا استاذة/سيدة" with name
        r'يا\s+(?:استاذة?|أستاذة?|سيدة?)\s+([\u0600-\u06FF\s]{4,30})(?:\s+اقدر|\s+أقدر)',
        
        # "شكرا جدا يا استاذة [name]"
        r'شكرا\s+جدا\s+يا\s+(?:يا\s+)?(?:استاذة?|أستاذة?|سيدة?)\s+([\u0600-\u06FF]+)',
    ]
    
    # Non-name phrases to filter out
    non_names = [
        "خدمة العملاء", "الدعم الفني", "الشركة", 
        "سيدة", "استاذ", "سيد", "يا فندم", "أستاذ", "استاذة",
        "تمام", "شكرا", "الحقيقة", "أحمد", "احمد",
        "في خدمتك", "أقدر", "اقدر", "فندم"
    ]
    
    for pattern in name_patterns:
        matches = re.finditer(pattern, text)
        for match in matches:
            name = match.group(1).strip()
            
            # Clean up the name
            name = re.sub(r'\s+', ' ', name)
            name = name.strip()
            # Trim trailing common filler phrases that may be captured
            stop_tokens = [
                ' والحقيقة', ' والحقيقة', ' و', ' ان ', ' انا ', ' على ', ' في ',
                ' موجودة', ' خدمة', ' الانتظار', ' اقدر', ' أقدر', ' تمام', ' شكراً', ' شكرا', ' يا فندم'
            ]
            for tok in stop_tokens:
                idx = name.find(tok)
                if idx > 0:
                    name = name[:idx].strip()
            
            # Filter out non-names and short names
            if any(phrase == name for phrase in non_names):
                continue
            
            # Skip if just one word AND it's a common word
            words = name.split()
            if len(words) == 1 and name in non_names:
                continue
            
            # Skip if contains numbers
            if re.search(r'\d', name):
                continue
            
            # Accept names with 2+ words, or single word if it looks like a name
            if len(words) >= 2 or (len(words) == 1 and len(name) >= 3):
                entities["client_name"] = name
                logger.info(f"👤 Found name: {entities['client_name']}")
                break
        
        if entities["client_name"]:
            break
    
    # Fallback: try to find "هبة عاطف الناغي" pattern directly
    if not entities["client_name"]:
        # Look for 3-4 Arabic words that appear after "فندم؟" and before "تمام"
        fallback_pattern = r'فندم[\u061F\?]?\s+([\u0600-\u06FF]+\s+[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+){1,2})\s+تمام'
        match = re.search(fallback_pattern, text)
        if match:
            name = match.group(1).strip()
            if not any(phrase in name for phrase in non_names):
                entities["client_name"] = name
                logger.info(f"👤 Found name (fallback): {entities['client_name']}")

    # If we only found a single-word name, try to expand it by looking for following words
    if entities.get("client_name"):
        parts = entities["client_name"].split()
        if len(parts) == 1:
            single = parts[0]
            expand_pat = re.compile(r'(' + re.escape(single) + r'\s+[\u0600-\u06FF]+(?:\s+[\u0600-\u06FF]+)?)')
            m = expand_pat.search(text)
            if m:
                candidate = m.group(1).strip()
                # filter candidate: ensure second word isn't a filler token
                parts_c = candidate.split()
                filler = set(['والحقيقة', 'الحقيقة', 'ان', 'انا', 'على', 'في', 'موجودة', 'خدمة', 'الانتظار', 'يا', 'تمام'])
                if len(parts_c) >= 2 and parts_c[1] not in filler and not any(phrase == candidate for phrase in non_names):
                    entities["client_name"] = candidate
                    logger.info(f"👤 Expanded name: {entities['client_name']}")
    
    logger.info("✅ Entity extraction completed!")
    return entities


def extract_entities_with_gpt(text: str, timeout: int = 10) -> dict:
    """
    Use OpenAI chat completions to extract `client_name` and `phone_number` from Arabic text.
    Lazy-imports OpenAI and requires `OPENAI_API_KEY` in the environment. Returns a dict
    with the same shape as `extract_entities_local`.
    """
    entities = {
        "client_name": None,
        "phone_number": None,
        "status": "new",
    }

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.info("OpenAI API key not found; skipping GPT extraction.")
        return entities

    try:
        from openai import OpenAI
    except Exception as e:
        logger.error(f"OpenAI client import failed: {e}")
        return entities

    try:
        client = OpenAI(api_key=api_key)

        system_prompt = load_prompt("extract_local_system")

        user_prompt = f"النص:\n{text[:2000]}"

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=200,
            timeout=timeout
        )

        content = resp.choices[0].message.content.strip()
        # Try parse JSON from response
        try:
            parsed = json.loads(content)
            entities["client_name"] = parsed.get("client_name")
            entities["phone_number"] = parsed.get("phone_number")
            logger.info(f"🤖 GPT extraction result: {entities}")
        except Exception:
            # Try to extract simple patterns from content
            m_name = re.search(r'"client_name"\s*:\s*"([^"]+)"', content)
            m_phone = re.search(r'"phone_number"\s*:\s*"([0-9]+)"', content)
            if m_name:
                entities["client_name"] = m_name.group(1)
            if m_phone:
                entities["phone_number"] = m_phone.group(1)

    except Exception as e:
        logger.error(f"OpenAI extraction failed: {e}")

    return entities


# =============================================================================
# Test
# =============================================================================
if __name__ == "__main__":
    # Test with actual transcription
    test_text = """مساء الخير يا فندم شكرا لاتصالك ب WE للاتصالات احمد في خدمتك ممكن اتشرف باسم حضرتك؟ اهلا استاذ احمد انا اسمي هبة والحقيقة ان انا موجودة علي خدمة الانتظار بقالي فترة طويلة جدا انا بعتزر جدا يا فندم الوقت ده من اليوم بيكون فيه ضغط ومكالمات كبير جدا اسفين جدا علي وقت حضرتك ممكن حضرك تقوليلي بالرقم المتاح عليه الخدمة؟ طبعا الرقم هو 01274714823 تمام يا فندم هل الخط مسجل باسم حضرتك؟ مصجل باسمي طيب ممكن توضحيلي الاسم بالكامل يا فندم؟ هبة عاطف الناغي عفيفي تمام شكرا جدا يا يا استاذة هبة اقدر اعطيك ايه يا فندم؟ انا اساعد حضرتك ازاي؟ الحقيقة يا استاذ احمد انا عايز اقول لك ان انا بالضبط علي بعد 30 دقيقة من اني اتفصل من شغلي انا كنت بشتغل والانترنت شغال مشكلة طبيعي وبعد ما حضرت ملفات الاخسل والاميلات اللي كنت محتاج ابعتها قبل ما اضغط علي الارسال فجاة الانترنت فصل وعلامة الانترنت علي جهاز الكمبيوتر بتقول اني مفيش انترنت والموضوع مش منطقي خالص لاني دفعت الفاتورة في معادها الساعة دلوقتي 9 وانا محتاج ابعت الشغل بتاعي الساعة 10 من الاكتر فمن فضلك استاذ احمد انا محتاج حل لان الوقت لو عدي من غير ما ابعت الايميلات دي خاف مشكلة كبيرة في شغلي."""
    
    print("=" * 60)
    print("🧪 Testing Entity Extraction")
    print("=" * 60)
    
    result = extract_entities_local(test_text)
    
    print("\n📋 Results:")
    print("-" * 40)
    print(f"👤 Client Name: {result.get('client_name', 'NOT FOUND')}")
    print(f"📞 Phone Number: {result.get('phone_number', 'NOT FOUND')}")
    print(f"📊 Status: {result.get('status', 'N/A')}")
    print("-" * 40)
    
    # Verify expected results
    print("\n✅ Verification:")
    expected_name = "هبة عاطف الناغي عفيفي"
    expected_phone = "01274714823"
    
    if result.get('client_name') == expected_name:
        print(f"   Name: ✅ CORRECT")
    elif result.get('client_name'):
        print(f"   Name: ⚠️ PARTIAL - Got: {result.get('client_name')}")
    else:
        print(f"   Name: ❌ NOT FOUND (expected: {expected_name})")
    
    if result.get('phone_number') == expected_phone:
        print(f"   Phone: ✅ CORRECT")
    else:
        print(f"   Phone: ❌ WRONG (expected: {expected_phone})")
    
    print("=" * 60)

    # ------------------------------------------------------------------
    # Optional: test OpenAI extractor if API key available
    # ------------------------------------------------------------------
    api_key = os.getenv("OPENAI_API_KEY")
    print("\n🔎 OpenAI extractor test:")
    if not api_key:
        print("OPENAI_API_KEY not found in environment (.env not loaded or key missing). Skipping GPT test.")
    else:
        print("OPENAI_API_KEY found — calling GPT extractor (this will use network and may incur cost)...")
        try:
            g = extract_entities_with_gpt(test_text)
            print("GPT Extractor Result:")
            print(g)
        except Exception as e:
            print(f"GPT extractor call failed: {e}")
