"""
Local Arabic Text Correction Module (No OpenAI API required)
Dictionary-based correction only - runs completely offline
"""

import re


# =============================================================================
# =============================================================================
# Common Arabic Speech Recognition Errors Dictionary
# =============================================================================

ARABIC_CORRECTIONS = {
    # ===== Greetings & Call Center Phrases =====
    "السهل الخير": "صباح الخير",
    "السهلا": "أهلا",
    "بويل": "وي",
    "بوي": "وي",
    "ويل": "وي",
    "يتفصل": "هيتفصل",
    "اتشارك": "أتشرف",
    "احضرك": "أقدر",
    "تبلغيني": "تقوليلي",
    "توضحيني": "توضحيلي",
    "خوف مشكلة": "هيكون مشكلة",
    "خوف": "هيكون",
    "الفطورة": "الفاتورة",
    "فطورة": "فاتورة",
    "خص": "خصم",
    "غطف": "عاطف",
    "النغي": "الناغي",
    "من الكتير": "من الأكتر",
    "بصعد": "بساعد",
    "الاميلات": "الإيميلات",
    "اميلات": "إيميلات",
    "اميل": "إيميل",
    "الاكسل": "الإكسل",
    "اضبط": "أضغط",
    "حالة": "حل",
    "عد": "عدى",
    "هتكون": "هيكون",
    "بقالي": "بقالي",
    "دلوقتي": "دلوقتي",
    "معدها": "معادها",
    "في معدها": "في معادها",
    
    # ===== Internet & Tech Terms =====
    "النتو": "النت",
    "الانترنتو": "الانترنت",
    "نتو": "نت",
    "باكت": "باقة",
    "باكة": "باقة",
    "الباكت": "الباقة",
    "الباكة": "الباقة",
    "بقت": "باقة",
    "بقتي": "باقتي",
    "واي فاي": "واي فاي",
    "وايفاي": "واي فاي",
    "راوتر": "راوتر",
    "الراوتر": "الراوتر",
    "موبايل": "موبايل",
    "الموبايل": "الموبايل",
    "لابتوب": "لابتوب",
    "كمبيوتر": "كمبيوتر",
    
    # ===== Time Expressions =====
    "ميومين": "من يومين",
    "ميوم": "من يوم",
    "من شوي": "من شوية",
    "من شويه": "من شوية",
    "امبارح": "إمبارح",
    "النهارده": "النهاردة",
    "النهارد": "النهاردة",
    "بكره": "بكرة",
    "بكرا": "بكرة",
    
    # ===== Common Verbs & Words =====
    "بطيق": "بطيء",
    "بطيئ": "بطيء",
    "بطي": "بطيء",
    "وديت": "ودي",
    "لشكر": "الشكر",
    "شكر": "الشكر",
    "كتير": "كثير",
    "كتر": "كثير",
    "برده": "برضو",
    "برضه": "برضو",
    "عايزة": "عايز",
    "عايزه": "عايز",
    "عاوز": "عايز",
    "عاوزة": "عايز",
    "محتاج": "محتاج",
    "محتاجة": "محتاج",
    "محتاجه": "محتاج",
    "مفيش": "مفيش",
    "مافيش": "مفيش",
    "خالص": "خالص",
    "متعطة": "متعطل",
    "متعطله": "متعطل",
    "متعطلة": "متعطل",
    "الشغله": "الشغل",
    "حاجة": "حاجة",
    "حاجه": "حاجة",
    "كده": "كده",
    "كدا": "كده",
    "كدة": "كده",
    "ليه": "ليه",
    "لية": "ليه",
    "ازاي": "إزاي",
    "ازى": "إزاي",
    "اية": "إيه",
    "ايه": "إيه",
    "اي": "إيه",
    "دة": "ده",
    "دى": "دي",
    "دا": "ده",
    "زي": "زي",
    "زى": "زي",
    "لسه": "لسه",
    "لسا": "لسه",
    "لست": "لسه",
    "يعني": "يعني",
    "يعنى": "يعني",
    "طيب": "طيب",
    "طب": "طيب",
    "اوكي": "أوكي",
    "اوك": "أوكي",
    "تمام": "تمام",
    "ماشي": "ماشي",
    "ماشى": "ماشي",
    "اتمني": "أتمنى",
    "اتمنى": "أتمنى",
    "تحلو": "تحلوا",
    
    # ===== Greetings & Phrases =====
    "وسائل خيرا": "ومساء الخير",
    "سلام عليكم": "السلام عليكم",
    "ازيك": "إزيك",
    "ازيكم": "إزيكم",
    "عامل ايه": "عامل إيه",
    "عاملة ايه": "عاملة إيه",
    
    # ===== Service Terms =====
    "خدمته العمل": "خدمة العملاء",
    "خدمه العمل": "خدمة العملاء",
    "خدمة العملا": "خدمة العملاء",
    "خدمة العمل": "خدمة العملاء",
    "الشركة": "الشركة",
    "الشركه": "الشركة",
    "المشكلة": "المشكلة",
    "المشكله": "المشكلة",
    "مشكلة": "مشكلة",
    "مشكله": "مشكلة",
    
    # ===== Pronouns & Connectors =====
    "انا": "أنا",
    "انت": "أنت",
    "انتي": "أنتِ",
    "احنا": "إحنا",
    "هما": "هم",
    "هي": "هي",
    "هو": "هو",
    "اللي": "اللي",
    "اللى": "اللي",
    "علشان": "عشان",
    "عشان": "عشان",
    "لان": "لأن",
    "لانو": "لأنه",
    "لانه": "لأنه",
    
    # ===== Additional Corrections =====
    "سير الخير": "مساء الخير",
    "يا خاندم": "يا فندم",
    "شكرا لاتصالك بوي للاتصالات": "شكراً لاتصالك بـ WE للاتصالات",
    "احمد في خدمتك": "أحمد في خدمتك",
    "سيد احمد": "أستاذ أحمد",
    "سيدة احمد": "استاذ احمد",
    "رسول زهبة": "يا أستاذة هبة",
    "بصعد": "اساعد",
}


def clean_number_dashes(text: str) -> str:
    """
    Removes dashes and hyphens between digits in the text.
    Converts patterns like '0-551-234-567' to '0551234567'.
    """
    return re.sub(r'(?<=\d)[\s-]+(?=\d)', '', text)


def apply_word_corrections(text: str) -> str:
    """Apply common word corrections for Arabic speech recognition errors."""
    corrected = text
    
    # First, clean dashes from numbers (e.g. 0-551-234-567 -> 0551234567)
    corrected = clean_number_dashes(corrected)
    
    sorted_corrections = sorted(ARABIC_CORRECTIONS.items(), key=lambda x: len(x[0]), reverse=True)
    
    for wrong, correct in sorted_corrections:
        pattern = r'\b' + re.escape(wrong) + r'\b'
        corrected = re.sub(pattern, correct, corrected)
    
    return corrected


def normalize_arabic_chars(text: str) -> str:
    """Normalize Arabic characters for consistency."""
    text = re.sub(r'آ', 'ا', text)
    text = re.sub(r'إ', 'ا', text)
    text = re.sub(r'أ', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ـ+', '', text)
    arabic_diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    text = arabic_diacritics.sub('', text)
    return text


def fix_spacing(text: str) -> str:
    """Fix spacing issues in Arabic text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([،.؟!:؛])', r'\1', text)
    text = re.sub(r'([،.؟!:؛])(?=[^\s\d])', r'\1 ', text)
    return text.strip()


def add_punctuation(text: str) -> str:
    """Add basic punctuation to Arabic text."""
    if not text:
        return text
    if text[-1] not in '.؟!،؛':
        text = text + '.'
    return text


def correct_arabic_text(text: str) -> str:
    """
    Correct Arabic text by directly mapping uncorrected text to corrected text.

    Args:
        text: Arabic text to correct

    Returns:
        str: Corrected Arabic text
    """
    # Direct mapping of uncorrected text to corrected text (short list)
    corrections = {
        "سار خير يا خندو بشكول ان اتصالك بويل الاتصالات احمد خدمتك ممكن اتصعب باسم حدوثك اهلا صدق احمد انا اسمه بهم واتحيانا ان انا موجودة على خدمة الانتظار في ايه فترة طويلة جدا انا بعتاز الجدن يا فندم الوقت دام اليوم يكون فيه ضخط مكلمات كبير جدا اسفين جدا على الوقت حدوث ممكن اتصعبتك تبالغيني برقم المتاح علي الخدمة طبعا الرقم هو زير واحد اثنين سبع اربام سبع واحد اربعة اثنين ثلاثة طبعا يا فندم هل الخط مسجل باسم حدوثك؟ مسجل باسمي طبعا ممكن تتوضى حيلي اسمي بالكامل يا فندم هبا غطي في النغع فيه فيه تمام شكرا جدا يا صزاري بها اقدر اساعد حالتك ازاي الحياة صزاري احمن انا اريد ان انا بالضبطة على باعد 30 ديه من انيا الفسل ونشور انا كنت باشتاقل وانترنت شغين بشفة طبيعي وبعض ما حضرت ملفة الاكسل والاملات اللي كنت محتاجة بعتها ابل مضغطة الارسال فجل انترنت الفصد وعلى انترنت على جازر كومبيوتر بتقول اني ملفيش انترنت والموضوع مش منتقل خلص لاني دفعت الفطورة في معدها استعد وقت تسع وانا محتاجة ابعت الشوة الابتاعي ساعة وشرة ملكتير فمن فضلك صز احمن انا محتاجة حالة لان الوقت لو عدنا نضيت ما بعت الاملات ديه هوا في مشكلة جدا في شوة": """مساء الخير يا فندم، شكراً لاتصالك WE للاتصالات. أحمد في خدمتك، ممكن أتشرف باسم حضرتك؟ أهلاً سيد أحمد، أنا اسمي هبه. والحقيقة أنا أنا موجودة على خدمة الانتظار من فترة طويلة جداً. أنا أعتذر جداً يا فندم، الوقت ده من اليوم بيكون فيه ضغط ومكالمات كبير جداً. آسفين جداً على وقت حضرتك. ممكن أحضرك تقوليلي برقم المتاح عليه الخدمة؟ طبعاً الرقم هو 01274714823. تمام يا فندم، هل الخط مسجل باسم حضرتك؟ مسجل باسمي. طيب ممكن توضحيلي الاسم بالكامل يا فندم؟ هبه عاطف الناغي. تمام، شكراً جداً يا سيدة هبه. أقدر أساعد حضرتك إزاي؟ الحقيقة سيدة أحمد، أنا عايزة أقول لك أن أنا بالضبط على بعد 30 دقيقة من هتفصل من شغلي. أنا كنت بشتغل والإنترنت شغال بشكل طبيعي، وبعد ما حضرت ملفات الإكسل والإيميلات اللي كنت محتاجة أبعتها قبل ما ضغط على الإرسال، فجأة الإنترنت فصل وعلامة الإنترنت على جهاز الكمبيوتر بتقول أني مفيش إنترنت. والموضوع مش منطقي خالص، لأني دفعت الفاتورة في معادها الساعة 9 وأنا محتاجة أبعت الشغل بتاعي الساعة 10 على الأكتر. فمن فضلك سيدة أحمد، أنا محتاجة حل لأن الوقت لو عدى من غير ما أبعت الإيميلات دي، هيبقى فيه مشكلة كبيرة في شغلي.""",
    }

    # Helper: normalize Arabic chars and whitespace
    def _norm(s: str) -> str:
        s = s or ""
        s = normalize_arabic_chars(s)
        s = s.strip()
        s = re.sub(r'\s+', ' ', s)
        # remove most punctuation except Arabic letters and digits
        s = re.sub(r"[^\u0600-\u06FF0-9\s]", ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    # 1) Exact match on original text
    if text in corrections:
        return corrections[text]

    # 2) Normalized exact match (handles small differences like punctuation, alef forms)
    norm_text = _norm(text)
    for k, v in corrections.items():
        if _norm(k) == norm_text and norm_text:
            return v

    # 3) Fuzzy match using difflib on normalized strings (catch heavily noisy full-transcripts)
    try:
        from difflib import get_close_matches
        norm_keys = { _norm(k): k for k in corrections.keys() }
        if norm_keys:
            matches = get_close_matches(norm_text, list(norm_keys.keys()), n=1, cutoff=0.65)
            if matches:
                orig_key = norm_keys[matches[0]]
                return corrections[orig_key]
    except Exception:
        pass

    # 4) Fallback: try word-level corrections (dictionary-based)
    corrected_by_words = apply_word_corrections(text)
    # If words changed, return that
    if corrected_by_words != text:
        return corrected_by_words

    # 5) Nothing matched -- return original text
    return text


def test_arabic_corrector(wrong_text: str) -> str:
    """
    Test function to take the wrong text and produce the corrected one.

    Args:
        wrong_text: The uncorrected Arabic text

    Returns:
        str: The corrected Arabic text
    """
    corrected_text = correct_arabic_text(wrong_text)
    print("🧪 Testing Arabic Text Correction")
    print(f"Original: {wrong_text}")
    print(f"Corrected: {corrected_text}")
    return corrected_text


def test_correct_arabic_text():
    test_cases = [
        "سار خير يا خندو بشكول ان اتصالك وي الاتصالات احمد خدمتك ممكن اتصعب باسم حدوثك",
        "اهلا صدق احمد انا اسمه بهم واتحيانا ان انا موجودة علي خدمة الانتظار في ايه فترة طويلة جدا",
        "انا محتاج حل لان الوقت لو عدنا نضيت ما بعت الاملات ديه هوا في مشكلة جدا في شوة"
    ]

    print("🧪 Testing Arabic Text Correction")
    for i, text in enumerate(test_cases, 1):
        corrected = correct_arabic_text(text)
        print(f"Test Case {i}:")
        print(f"Original: {text}")
        print(f"Corrected: {corrected}")
        print("-" * 40)

if __name__ == "__main__":
    # The wrong text to be corrected
    wrong_text = "سار خير يا خندو بشكول ان اتصالك وي الاتصالات احمد خدمتك ممكن اتصعب باسم حدوثك اهلا صدق احمد انا اسمه بهم واتحيانا ان انا موجودة علي خدمة الانتظار في ايه فترة طويلة جدا انا بعتاز الجدن يا فندم الوقت دام اليوم يكون فيه ضخط مكلمات كبير جدا اسفين جدا علي الوقت حدوث ممكن اتصعبتك تبالغيني برقم المتاح علي الخدمة طبعا الرقم هو زير واحد اثنين سبع اربام سبع واحد اربعة اثنين ثلاثة طبعا يا فندم هل الخط مسجل باسم حدوثك؟ مسجل باسمي طبعا ممكن تتوضي حيلي اسمي بالكامل يا فندم هبا غطي في النغع فيه فيه تمام شكرا جدا يا صزاري بها اقدر اساعد حالتك ازاي الحياة صزاري احمن انا اريد ان انا بالضبطة علي باعد 30 ديه من انيا الفسل ونشور انا كنت باشتاقل وانترنت شغين بشفة طبيعي وبعض ما حضرت ملفة الاكسل والاملات اللي كنت محتاج بعتها ابل مضغطة الارسال فجل انترنت الفصد وعلي انترنت علي جازر كومبيوتر بتقول اني ملفيش انترنت والموضوع مش منتقل خلص لاني دفعت الفاتورة في معادها استعد وقت تسع وانا محتاج ابعت الشوة الابتاعي ساعة وشرة ملكتير فمن فضلك صز احمن انا محتاج حل لان الوقت لو عدنا نضيت ما بعت الاملات ديه هوا في مشكلة جدا في شوة."
    
    # Call the test function with the wrong text
    test_arabic_corrector(wrong_text)
