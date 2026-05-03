# -*- coding: utf-8 -*-
"""
Simple test for LLM_turns diarization — raw text mode.

Sends a plain Arabic call center transcription (no timestamps/segments)
to diarize_with_llm() and validates the output.
"""

import json
import logging

from LLM_turns import diarize_with_llm

# Enable logging so we can see API attempts
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

# ---------- sample raw text (no segments, no timestamps) ----------
# This is the EXACT text that was producing incorrect results
sample_text = """
مساء الخير يا فندم شكرا لاتصالك ب WE للاتصالات احمد في خدمتك ممكن اتشرف باسم حضرتك؟ اهلا استاذ احمد انا اسمي هبة والحقيقة ان انا موجودة علي خدمة الانتظار بقالي فترة طويلة جدا انا بعتذر جدا يا فندم الوقت ده من اليوم بيكون فيه ضغط ومكالمات كبير جدا اسفين جدا علي وقت حضرتك ممكن حضرتك تبلغيني بالرقم المتاح عليه الخدمة؟ طبعا الرقم هو 01274714823 تمام يا فندم هل الخط مسجل باسم حضرتك؟ مسجل باسمي طيب ممكن توضحي لي الاسم بالكامل يا فندم؟ هبة عاطف الناغي عفيفي تمام شكرا جدا يا يا استاذة هبة اقدر اساعدك في ايه يا فندم؟ انا اساعد حضرتك ازاي؟ الحقيقة يا استاذ احمد انا عايز اقول لك ان انا بالضبط علي بعد 30 دقيقة من اني اتفصل من شغلي انا كنت بشتغل والانترنت شغال مشكلة طبيعي وبعد ما حضرت ملفات الاكسل والاميلات اللي كنت محتاج ابعتها قبل ما اضبط علي الارسال فجاة الانترنت فصل وعلامة الانترنت علي جهاز الكمبيوتر بتقول ان مفيش انترنت والموضوع مش منطقي خالص لاني دفعت الفاتورة في معادها الساعة دلوقتي 9 وانا محتاج ابعت الشغل بتاعي الساعة 10 من الكتير فمن فضلك استاذ احمد انا محتاج حالة لان الوقت لو عد من غير ما ابعت الاميلات دي هتكون مشكلة كبيرة في شغلي.
"""


# Expected speaker sequence (for validation):
# 1. Agent: "مساء الخير يا فندم شكرا لاتصالك ب WE للاتصالات احمد في خدمتك ممكن اتشرف باسم حضرتك؟"
# 2. Customer: "اهلا استاذ احمد انا اسمي هبة..."
# 3. Agent: "انا بعتذر جدا يا فندم..."
# 4. Customer: "طبعا الرقم هو 01274714823"
# 5. Agent: "تمام يا فندم هل الخط مسجل باسم حضرتك؟"
# 6. Customer: "مسجل باسمي"
# 7. Agent: "طيب ممكن توضحي لي الاسم بالكامل يا فندم؟"
# 8. Customer: "هبة عاطف الناغي عفيفي"
# 9. Agent: "تمام شكرا جدا يا يا استاذة هبة اقدر اساعدك في ايه يا فندم؟ انا اساعد حضرتك ازاي؟"
# 10. Customer: "الحقيقة يا استاذ احمد..." (long problem description)


def test_raw_text_diarization():
    """Test that diarize_with_llm works with a plain text string."""
    print("=" * 70)
    print("TEST: Raw text diarization (no segments/timestamps)")
    print("Testing the EXACT text that had errors before")
    print("=" * 70)

    result = diarize_with_llm(sample_text)

    # ---- basic assertions ----
    assert result is not None, "diarize_with_llm returned None — API call likely failed"
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) > 0, "Result is an empty list"

    # ---- structure checks ----
    speakers_found = set()
    for i, turn in enumerate(result):
        assert "speaker" in turn, f"Turn {i} missing 'speaker' key"
        assert "text" in turn, f"Turn {i} missing 'text' key"
        assert turn["speaker"] in {"Agent", "Customer", "Unknown"}, (
            f"Turn {i} has invalid speaker: '{turn['speaker']}'"
        )
        assert len(turn["text"].strip()) > 0, f"Turn {i} has empty text"
        speakers_found.add(turn["speaker"])

    # ---- content checks ----
    assert "Agent" in speakers_found, "No 'Agent' speaker found in result"
    assert "Customer" in speakers_found, "No 'Customer' speaker found in result"

    # The first turn should be the Agent (greeting)
    assert result[0]["speaker"] == "Agent", (
        f"Expected first speaker to be Agent, got '{result[0]['speaker']}'"
    )

    print("\n✅ All assertions passed!\n")
    print(f"Total turns identified: {len(result)}")
    print(f"Speakers found: {speakers_found}")
    
    # ---- CRITICAL VALIDATION: Check for previously misclassified segments ----
    print("\n" + "=" * 70)
    print("🔍 VALIDATING CRITICAL SEGMENTS (previously had errors)")
    print("=" * 70)
    
    # Find segments with key phrases
    errors = []
    
    for i, turn in enumerate(result, 1):
        text = turn["text"]
        speaker = turn["speaker"]
        
        # Test Case 1: "أقدر أساعدك" or "أنا أساعد حضرتك" should be Agent
        if "أساعد" in text and "حضرتك" in text:
            if speaker != "Agent":
                errors.append(f"❌ Turn {i}: '{text[:50]}...' should be AGENT (offering help), but got {speaker}")
            else:
                print(f"✅ Turn {i}: Correctly identified AGENT offering help")
        
        # Test Case 2: "يا أستاذ أحمد" or "يا استاذ احمد" should be Customer
        if "استاذ احمد" in text.lower() or "أستاذ أحمد" in text:
            if speaker != "Customer":
                errors.append(f"❌ Turn {i}: '{text[:50]}...' should be CUSTOMER (addressing Agent Ahmed), but got {speaker}")
            else:
                print(f"✅ Turn {i}: Correctly identified CUSTOMER addressing Ahmed")
        
        # Test Case 3: Long problem description should be Customer
        if len(text) > 200 and ("الانترنت" in text or "الانترنت" in text.replace("الأنترنت", "الانترنت")):
            if speaker != "Customer":
                errors.append(f"❌ Turn {i}: Long internet problem description should be CUSTOMER, but got {speaker}")
            else:
                print(f"✅ Turn {i}: Correctly identified CUSTOMER explaining problem")
    
    print("\n--- Full Diarization Result ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # Report errors if any
    if errors:
        print("\n" + "=" * 70)
        print("⚠️  ERRORS FOUND:")
        print("=" * 70)
        for error in errors:
            print(error)
        raise AssertionError(f"Found {len(errors)} classification errors. See details above.")
    else:
        print("\n" + "=" * 70)
        print("🎉 ALL CRITICAL SEGMENTS CORRECTLY CLASSIFIED!")
        print("=" * 70)


def test_billing_complaint_diarization():
    """Test diarization with a billing complaint conversation."""
    print("\n" + "=" * 70)
    print("TEST: Billing Complaint Diarization")
    print("Testing conversation between Yasmin (Agent) and Mahmoud (Customer)")
    print("=" * 70)

    billing_sample = """
مساء الخير يا فندم شكرا لاتصالك مع حضرتك ياسمين ممكن اعرف اسم حضرتك؟
مساء النور يا استاذة ياسمين مع حضرتك محمود سامي.
اهلا يا استاذ محمود اتفضل اقدر اساعد حضرتك في ايه؟
الحقيقة يا فندم فاتورة الشهر ده عالية جدا عن المعتاد وانا مش فاهم السبب خالص.
تمام يا فندم قيمة الفاتورة وصلت لكام؟
وصلت ل 520 جنيه مع اني عمري ما بدفع اكتر من 300 جنيه.
تمام يا فندم لحظة واحدة هراجع البيانات...
"""

    result = diarize_with_llm(billing_sample)

    # ---- basic assertions ----
    assert result is not None, "diarize_with_llm returned None — API call likely failed"
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) > 0, "Result is an empty list"

    # ---- structure checks ----
    speakers_found = set()
    for i, turn in enumerate(result):
        assert "speaker" in turn, f"Turn {i} missing 'speaker' key"
        assert "text" in turn, f"Turn {i} missing 'text' key"
        assert turn["speaker"] in {"Agent", "Customer", "Unknown"}, (
            f"Turn {i} has invalid speaker: '{turn['speaker']}'"
        )
        assert len(turn["text"].strip()) > 0, f"Turn {i} has empty text"
        speakers_found.add(turn["speaker"])

    # ---- content checks ----
    assert "Agent" in speakers_found, "No 'Agent' speaker found in result"
    assert "Customer" in speakers_found, "No 'Customer' speaker found in result"

    # The first turn should be the Agent (Yasmin greeting)
    assert result[0]["speaker"] == "Agent", (
        f"Expected first speaker to be Agent, got '{result[0]['speaker']}'"
    )

    print("\n✅ All basic assertions passed!\n")
    print(f"Total turns identified: {len(result)}")
    print(f"Speakers found: {speakers_found}")

    # ---- VALIDATION: Check key segments ----
    print("\n" + "=" * 70)
    print("🔍 VALIDATING KEY SEGMENTS")
    print("=" * 70)

    errors = []
    
    for i, turn in enumerate(result, 1):
        text = turn["text"]
        speaker = turn["speaker"]
        
        # Agent Yasmin greeting (first turn)
        if "ياسمين" in text and "اتصالك" in text:
            if speaker != "Agent":
                errors.append(f"❌ Turn {i}: Agent greeting with name Yasmin should be AGENT, but got {speaker}")
            else:
                print(f"✅ Turn {i}: Correctly identified AGENT (Yasmin) greeting")
        
        # Customer introducing himself (Mahmoud Sami)
        if "محمود سامي" in text or "محمود" in text and "مساء النور" in text:
            if speaker != "Customer":
                errors.append(f"❌ Turn {i}: Customer introducing as Mahmoud should be CUSTOMER, but got {speaker}")
            else:
                print(f"✅ Turn {i}: Correctly identified CUSTOMER (Mahmoud) introduction")
        
        # Agent offering help
        if "اقدر اساعد" in text and "حضرتك" in text and "استاذ محمود" in text:
            if speaker != "Agent":
                errors.append(f"❌ Turn {i}: Agent offering help should be AGENT, but got {speaker}")
            else:
                print(f"✅ Turn {i}: Correctly identified AGENT offering help")
        
        # Customer complaint about bill
        if "فاتورة" in text and ("عالية" in text or "520" in text):
            if speaker != "Customer":
                errors.append(f"❌ Turn {i}: Customer complaining about bill should be CUSTOMER, but got {speaker}")
            else:
                print(f"✅ Turn {i}: Correctly identified CUSTOMER complaint")
        
        # Agent asking for bill amount
        if "القيمة" in text or ("وصلت لكام" in text or "الفاتورة" in text):
            if "فاتورة الشهر ده عالية" not in text:  # Skip customer's complaint
                if speaker == "Agent":
                    print(f"✅ Turn {i}: Correctly identified AGENT asking about bill")

    print("\n--- Full Diarization Result ---")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Report errors if any
    if errors:
        print("\n" + "=" * 70)
        print("⚠️  ERRORS FOUND:")
        print("=" * 70)
        for error in errors:
            print(error)
        raise AssertionError(f"Found {len(errors)} classification errors. See details above.")
    else:
        print("\n" + "=" * 70)
        print("🎉 ALL KEY SEGMENTS CORRECTLY CLASSIFIED!")
        print("=" * 70)


if __name__ == "__main__":
    # Run both test cases
    test_raw_text_diarization()
    test_billing_complaint_diarization()
