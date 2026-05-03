import re
import os
import logging
import sys

# Ensure Project root is in path for prompt_loader
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from prompt_loader import load_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Lazy initialization of OpenAI client
client = None

def get_openai_client():
    global client
    if client is None:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return client


class EntityExtractor:
    def __init__(self):
        # ---------- Phone number patterns (Egypt + Saudi) ----------
        self.phone_patterns = [
            r'\b01[0-2]\d{8}\b',
            r'\b02\d{8}\b',
            r'\b03\d{8}\b',
            # Saudi mobile numbers (05x)
            r'\b05[0-9]\d{7}\b',
            # Numbers with dashes/spaces: 0-551-234-567, 055-123-4567, etc.
            r'\b0[\s-]?5[0-9][\s-]?\d{1,4}[\s-]?\d{1,4}[\s-]?\d{1,4}\b',
            r'\b01[0-2][\s-]?\d{1,4}[\s-]?\d{1,4}[\s-]?\d{1,4}\b',
        ]

        # ---------- Name extraction patterns ----------
        self.strict_name_patterns = [
            # Most important: "توضحي لي الاسم بالكامل يا فندم؟ [NAME] تمام"
            r'توضحي?\s+لي\s+الاسم\s+بالكامل\s+يا\s+فندم\??\s*([\u0600-\u06FF\s]+?)(?:\s+تمام|\s+شكرا|$)',
            # "ممكن توضحي لي الاسم بالكامل يا فندم؟ [NAME]"
            r'ممكن\s+توضحي?\s+لي\s+الاسم\s+بالكامل\s+يا\s+فندم\??\s*([\u0600-\u06FF\s]+?)(?:\s+تمام|\s+شكرا|$)',
            # "الاسم بالكامل يا فندم؟ [NAME]"
            r'الاسم\s+بالكامل\s+يا\s+فندم\??\s*([\u0600-\u06FF\s]+?)(?:\s+تمام|\s+شكرا|$)',
            # "اسم حضرتك بالكامل؟ [NAME]"
            r'اسم\s+حضرتك\s+بالكامل\??\s*([\u0600-\u06FF\s]+?)(?:\s+تمام|\s+شكرا|$)',
        ]

        # ---------- Stop words that indicate name has ended ----------
        self.name_stop_words = [
            'ورقم', 'رقم', 'جوالي', 'تليفوني', 'تلفوني', 'موبايلي', 'هاتفي',
            'وأنا', 'وانا', 'أتمنى', 'اتمنى', 'ومشكلتي', 'وعندي', 'والمشكلة',
            'ومحتاج', 'وياريت', 'وأرجو', 'وارجو', 'عندي', 'محتاج',
            'واريد', 'وأريد', 'بتاعي', 'الخاص', 'ورقمي'
        ]

        self.soft_name_patterns = [
            # Pattern 1: After asking for full name
            r'الاسم\s+بالكامل\s+يا\s+فندم\s*([\u0600-\u06FF\s]{4,}?)(?:\s+تمام|\s+شكرا|$)',
            # Pattern 2: "أنا اسمي X" - more specific capture
            r'انا\s+اسمي\s+([\u0600-\u06FF\s]{4,}?)(?:\s+و|$)',
            # Pattern 3: Exact full name response pattern
            r'(?:مسجل\s+باسمي|اسمي)\s*([\u0600-\u06FF]{2,}\s+[\u0600-\u06FF]{2,}(?:\s+[\u0600-\u06FF]{2,})*?)(?:\s+تمام|\s+شكرا|\s+احمد|$)',
            # Pattern 4: Name after greetings
            r'(?:اهلا|أهلا)\s+(?:استاذ|سيد)\s+\w+\s+انا\s+اسمي\s+([\u0600-\u06FF\s]{4,}?)(?:\s+و|$)',
            # معاك / معك + اسم العميل
            r'معاك\s+([\u0600-\u06FF\s]{4,}?)(?:\s+و|$)',
            r'معك\s+([\u0600-\u06FF\s]{4,}?)(?:\s+و|$)',
            # Pattern 6: "يا استاذة [AGENT] مع حضرتك [CLIENT_NAME]" - works on normalized text (no punctuation)
            r'يا\s+(?:استاذ[ةه]?|أستاذ[ةه]?)\s+[\u0600-\u06FF]+\s+مع\s+حضرتك\s+([\u0600-\u06FF]+\s+[\u0600-\u06FF]+)(?:\s+اهلا|\s*$)',
            # Pattern 7: "مساء النور يا استاذة X مع حضرتك [NAME]" - billing greeting
            r'مساء\s+النور\s+يا\s+(?:استاذ[ةه]?|أستاذ[ةه]?)\s+[\u0600-\u06FF]+\s+مع\s+حضرتك\s+([\u0600-\u06FF]+\s+[\u0600-\u06FF]+)(?:\s|$)',
        ]

        # ---------- Agent name extraction patterns ----------
        # NOTE: Agent patterns should match the FIRST speaker (agent greeting)
        self.agent_name_patterns = [
            r'([\u0600-\u06FF]+)\s+في\s+خدمتك',
            r'أنا\s+([\u0600-\u06FF]+)\s+من\s+خدمة',
            r'معاك\s+([\u0600-\u06FF]+)\s+من',
            # REMOVED: r'أهلا[ً]?\s+(?:يا\s+)?(?:أستاذ|استاذ|سيد|سيدة)\s+...' - this was matching CUSTOMER names!
            r'شكرا[ً]?\s+جدا[ً]?\s+يا\s+استاذة\s+([\u0600-\u06FF]+)\s+اقدر',  # thanking agent by name
            # Agent greeting: "مع حضرتك [AGENT_NAME] ممكن اعرف" (at start of call)
            r'(?:شكرا\s+)?(?:على\s+)?(?:ل)?اتصالك\s+مع\s+حضرتك\s+([\u0600-\u06FF]+)\s+ممكن',  # handles "شكرا على اتصالك" or "شكرا لاتصالك"
            r'مع\s+حضرتك\s+([\u0600-\u06FF]+)\s+ممكن\s+(?:اعرف|أعرف)',
            r'اتصالك\s+(?:مع|ب)\s*حضرتك\s+([\u0600-\u06FF]+)',  # fallback: "اتصالك مع حضرتك [NAME]"
        ]

        # ---------- Job titles to filter out from names ----------
        self.job_titles = [
            "خدمة العملاء",
            "الدعم الفني",
            "الاتصالات",
            "الشركة"
        ]

    # ---------- Text normalization ----------
    def normalize_text(self, text):
        if not text:
            return ""
        text = text.strip()
        # Remove non-Arabic, non-digit, non-space characters
        # Also remove Arabic punctuation marks (U+0600-U+060F are punctuation/formatting)
        text = re.sub(r'[^\u0621-\u064A\u0660-\u0669\u06F0-\u06F90-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text

    # ---------- Phone ----------
    def extract_phone_number(self, text):
        # First try with original text
        for pattern in self.phone_patterns:
            match = re.search(pattern, text)
            if match:
                # Clean the matched number: remove dashes and spaces
                phone = re.sub(r'[\s-]', '', match.group())
                return phone
        
        # Try after stripping dashes and extra spaces from text
        cleaned_text = re.sub(r'(?<=\d)[\s-]+(?=\d)', '', text)
        for pattern in self.phone_patterns:
            match = re.search(pattern, cleaned_text)
            if match:
                phone = re.sub(r'[\s-]', '', match.group())
                return phone
        
        return None

    # ---------- Client Name ----------
    def extract_client_name(self, text):
        text = self.normalize_text(text)
        logger.debug(f"Extracting name from normalized text: {text[:100]}...")

        strict_names = []
        soft_names = []

        # Try strict patterns first
        for pattern in self.strict_name_patterns:
            match = re.search(pattern, text)
            if match:
                name = self.clean_name(match.group(1))
                if name:
                    logger.info(f"Found client name (strict pattern): {name}")
                    strict_names.append(name)

        # Then soft patterns
        for pattern in self.soft_name_patterns:
            match = re.search(pattern, text)
            if match:
                name = self.clean_name(match.group(1))
                if name:
                    logger.info(f"Found client name (soft pattern): {name}")
                    soft_names.append(name)

        # Prefer strict patterns over soft patterns
        candidates = strict_names if strict_names else soft_names
        
        if candidates:
            # Choose the name with 2-4 words (typical Arabic name)
            # If multiple candidates exist, prefer the one with 3-4 words
            def score_name(name):
                words = name.split()
                word_count = len(words)
                # Score based on word count: 2-4 words are ideal
                if 2 <= word_count <= 4:
                    return (2, word_count, len(name))  # Preferred range
                else:
                    return (1, abs(word_count - 3), len(name))  # Not ideal
            
            best_name = max(candidates, key=score_name)
            logger.info(f"Selected best name from {len(candidates)} candidates: {best_name}")
            return best_name

        logger.warning("No client name found in text")
        return None

    def clean_name(self, name):
        name = name.strip()
        
        # Remove leading and trailing punctuation/non-Arabic characters
        # Keep only Arabic letters and spaces
        name = re.sub(r'^[^ء-ي]+', '', name).strip()  # Remove leading non-Arabic
        name = re.sub(r'[^ء-ي\s]+$', '', name).strip()  # Remove trailing non-Arabic
        
        # Remove Arabic diacritics if present
        name = re.sub(r'[\u064B-\u0652]', '', name)
        
        # Truncate at stop words (e.g., "ورقم جوالي" should not be part of name)
        for stop_word in self.name_stop_words:
            idx = name.find(stop_word)
            if idx > 0:
                name = name[:idx].strip()
        
        # Remove trailing conjunctions/articles
        name = re.sub(r'\s+(?:و|و|ف|ب|ل|ال)$', '', name.strip())
        
        words = name.split()

        # Must have at least 2 words (first and last name)
        if len(words) < 2:
            return None
        
        # Keep at most 5 words (typically Arabic names have 2-4 words)
        if len(words) > 5:
            words = words[:5]
        
        name = ' '.join(words)

        # Check if any job title is in the name
        if any(title in name for title in self.job_titles):
            return None

        return name

    # ---------- Agent Name ----------
    def extract_agent_name(self, text):
        """Extract the customer service agent's name from the transcription"""
        text = self.normalize_text(text)
        
        for pattern in self.agent_name_patterns:
            match = re.search(pattern, text)
            if match:
                agent_name = match.group(1).strip()
                # Remove any trailing punctuation or special characters
                agent_name = re.sub(r'[^\u0600-\u06FF]+$', '', agent_name).strip()
                # Filter out common words that aren't names
                if agent_name and len(agent_name) >= 2 and agent_name not in ['يا', 'في', 'من', 'الى', 'على']:
                    logger.info(f"Found agent name: {agent_name}")
                    return agent_name
        
        logger.warning("No agent name found in text")
        return None

    # ---------- Problem Statement ----------
    def extract_problem_statement(self, text):
        """Extract the customer's problem/complaint from the text"""
        text_normalized = self.normalize_text(text)
        
        # Look for complaint indicators
        problem_indicators = [
            r'(?:مشكلتي|المشكلة|عندي مشكلة|شكوى|بشتكي|عايز اشتكي)\s+([\u0600-\u06FF\s]{10,})',
            r'(?:محتاج|عايز|ابغى|ابي|اريد|أريد)\s+([\u0600-\u06FF\s]{10,})',
            r'(?:النت|الانترنت|الخدمة|الفاتورة|الباقة)\s+([\u0600-\u06FF\s]{10,})',
        ]
        
        for pattern in problem_indicators:
            match = re.search(pattern, text_normalized)
            if match:
                problem = match.group(0).strip()
                # Limit to reasonable length
                words = problem.split()
                if len(words) > 30:
                    problem = ' '.join(words[:30])
                return problem
        
        # Fallback: return the full text as the problem statement
        return text

    # ---------- Main ----------
    def extract_entities(self, text):
        return {
            "client_name": self.extract_client_name(text),
            "phone_number": self.extract_phone_number(text),
            "agent_name": self.extract_agent_name(text)
        }


# ---------- Standalone function ----------
def extract_call_center_entities(text):
    extractor = EntityExtractor()
    return extractor.extract_entities(text)


# ---------- GPT-based Entity Extraction ----------
def extract_entities_with_gpt(text):
    """
    Extract entities from Arabic call center transcription using GPT.
    More accurate than regex-based extraction.
    """
    try:
        logger.info("🤖 Extracting entities with GPT...")
        
        prompt = load_prompt("extract_entities_gpt_user", text=text)

        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": load_prompt("extract_entities_gpt_system")},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        result_text = response.choices[0].message.content.strip()
        logger.info(f"GPT extraction result: {result_text}")
        
        # Parse the response
        entities = {
            "client_name": None,
            "phone_number": None,
            "status": "new"
        }
        
        for line in result_text.split('\n'):
            line = line.strip()
            if line.startswith('اسم_العميل:'):
                value = line.replace('اسم_العميل:', '').strip()
                if value and value != "غير متوفر":
                    entities["client_name"] = value
            elif line.startswith('رقم_الهاتف:'):
                value = line.replace('رقم_الهاتف:', '').strip()
                if value and value != "غير متوفر":
                    entities["phone_number"] = value
        
        logger.info("✅ GPT entity extraction completed!")
        return entities
        
    except Exception as e:
        logger.error(f"GPT extraction error: {e}")
        # Fallback to regex-based extraction
        extractor = EntityExtractor()
        return extractor.extract_entities(text)


# ---------- Test ----------
if __name__ == "__main__":
    # Use the corrected text for testing
    corrected_text = """مساء الخير يا فندم، شكراً لاتصالك WE للاتصالات. أحمد في خدمتك، ممكن أتشرف باسم حضرتك؟ أهلاً سيد أحمد، أنا اسمي هبه. والحقيقة أنا أنا موجودة على خدمة الانتظار من فترة طويلة جداً. أنا أعتذر جداً يا فندم، الوقت ده من اليوم بيكون فيه ضغط ومكالمات كبير جداً. آسفين جداً على وقت حضرتك. ممكن أحضرك تقوليلي برقم المتاح عليه الخدمة؟ طبعاً الرقم هو 01274714823. تمام يا فندم، هل الخط مسجل باسم حضرتك؟ مسجل باسمي. طيب ممكن توضحيلي الاسم بالكامل يا فندم؟ هبه عاطف الناغي. تمام، شكراً جداً يا سيدة هبه. أقدر أساعد حضرتك إزاي؟ الحقيقة سيدة أحمد، أنا عايزة أقول لك أن أنا بالضبط على بعد 30 دقيقة من هتفصل من شغلي. أنا كنت بشتغل والإنترنت شغال بشكل طبيعي، وبعد ما حضرت ملفات الإكسل والإيميلات اللي كنت محتاجة أبعتها قبل ما ضغط على الإرسال، فجأة الإنترنت فصل وعلامة الإنترنت على جهاز الكمبيوتر بتقول أني مفيش إنترنت. والموضوع مش منطقي خالص، لأني دفعت الفاتورة في معادها الساعة 9 وأنا محتاجة أبعت الشغل بتاعي الساعة 10 على الأكتر. فمن فضلك سيدة أحمد، أنا محتاجة حل لأن الوقت لو عدى من غير ما أبعت الإيميلات دي، هيبقى فيه مشكلة كبيرة في شغلي."""

    extractor = EntityExtractor()
    result = extractor.extract_entities(corrected_text)

    print("الكيانات المستخرجة:")
    for k, v in result.items():
        print(f"{k}: {v}")

    # Test cases for name and phone extraction
    def test_name_extraction():
        name = extractor.extract_client_name(corrected_text)
        expected_name = "هبه عاطف الناغي"
        assert name == expected_name, f"Name extraction failed: expected '{expected_name}', got '{name}'"
        print(f"✅ Name extraction test passed: '{name}'")

    def test_phone_extraction():
        phone = extractor.extract_phone_number(corrected_text)
        expected_phone = "01274714823"
        assert phone == expected_phone, f"Phone extraction failed: expected '{expected_phone}', got '{phone}'"
        print(f"✅ Phone extraction test passed: '{phone}'")

    # Run tests
    test_name_extraction()
    test_phone_extraction()
    print("✅ All extraction tests passed!")
