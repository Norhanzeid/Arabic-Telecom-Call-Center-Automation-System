#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test for client name extraction
"""

from modules.speech_to_text.extract import EntityExtractor

# Test with the exact transcription
sample_text = """
مساء الخير يا فندم شكرا لاتصالك ب WE للاتصالات احمد في خدمتك ممكن اتشرف باسم حضرتك؟ اهلا استاذ احمد انا اسمي هبة والحقيقة ان انا موجودة علي خدمة الانتظار بقالي فترة طويلة جدا انا بعتذر جدا يا فندم الوقت ده من اليوم بيكون فيه ضغط ومكالمات كبير جدا اسفين جدا علي وقت حضرتك ممكن حضرتك تبلغيني بالرقم المتاح عليه الخدمة؟ طبعا الرقم هو 01274714823 تمام يا فندم هل الخط مسجل باسم حضرتك؟ مسجل باسمي طيب ممكن توضحي لي الاسم بالكامل يا فندم؟ هبة عاطف الناغي عفيفي تمام شكرا جدا يا يا استاذة هبة اقدر اساعدك في ايه يا فندم؟ انا اساعد حضرتك ازاي؟ الحقيقة يا استاذ احمد انا عايز اقول لك ان انا بالضبط علي بعد 30 دقيقة من اني اتفصل من شغلي انا كنت بشتغل والانترنت شغال مشكلة طبيعي وبعد ما حضرت ملفات الاكسل والاميلات اللي كنت محتاج ابعتها قبل ما اضبط علي الارسال فجاة الانترنت فصل وعلامة الانترنت علي جهاز الكمبيوتر بتقول ان مفيش انترنت والموضوع مش منطقي خالص لاني دفعت الفاتورة في معادها الساعة دلوقتي 9 وانا محتاج ابعت الشغل بتاعي الساعة 10 من الكتير فمن فضلك استاذ احمد انا محتاج حالة لان الوقت لو عد من غير ما ابعت الاميلات دي هتكون مشكلة كبيرة في شغلي.
"""

print("=" * 70)
print("CLIENT NAME EXTRACTION TEST")
print("=" * 70)

extractor = EntityExtractor()

print("\nTesting name extraction...")
name = extractor.extract_client_name(sample_text)
phone = extractor.extract_phone_number(sample_text)

print("\n" + "=" * 70)
print("RESULTS:")
print("=" * 70)
print("Client Name: ", name)
print("Phone Number:", phone)
print("=" * 70)

# Simple validation
if name and "هبة" in name:
    print("\nName extraction looks GOOD!")
    print("Expected: هبة عاطف الناغي عفيفي")
    print("Got:      " + name)
else:
    print("\nName extraction needs improvement")
    
if phone == "01274714823":
    print("Phone extraction is CORRECT!")
else:
    print("Phone extraction issue: got '" + str(phone) + "'")
