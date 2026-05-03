"""
Test script to debug entity extraction
"""

from modules.speech_to_text.extract import extract_call_center_entities
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Test with sample transcription
sample_text = """
مساء الخير يا فندم شكرا لاتصالك ب WE للاتصالات احمد في خدمتك ممكن اتشرف باسم حضرتك؟ اهلا استاذ احمد انا اسمي هبة والحقيقة ان انا موجودة علي خدمة الانتظار بقالي فترة طويلة جدا انا بعتذر جدا يا فندم الوقت ده من اليوم بيكون فيه ضغط ومكالمات كبير جدا اسفين جدا علي وقت حضرتك ممكن حضرتك تبلغيني بالرقم المتاح عليه الخدمة؟ طبعا الرقم هو 01274714823 تمام يا فندم هل الخط مسجل باسم حضرتك؟ مسجل باسمي طيب ممكن توضحي لي الاسم بالكامل يا فندم؟ هبة عاطف الناغي عفيفي تمام شكرا جدا يا يا استاذة هبة اقدر اساعدك في ايه يا فندم؟ انا اساعد حضرتك ازاي؟ الحقيقة يا استاذ احمد انا عايز اقول لك ان انا بالضبط علي بعد 30 دقيقة من اني اتفصل من شغلي انا كنت بشتغل والانترنت شغال مشكلة طبيعي وبعد ما حضرت ملفات الاكسل والاميلات اللي كنت محتاج ابعتها قبل ما اضبط علي الارسال فجاة الانترنت فصل وعلامة الانترنت علي جهاز الكمبيوتر بتقول ان مفيش انترنت والموضوع مش منطقي خالص لاني دفعت الفاتورة في معادها الساعة دلوقتي 9 وانا محتاج ابعت الشغل بتاعي الساعة 10 من الكتير فمن فضلك استاذ احمد انا محتاج حالة لان الوقت لو عد من غير ما ابعت الاميلات دي هتكون مشكلة كبيرة في شغلي.
"""

print("="*60)
print("Testing Entity Extraction")
print("="*60)
print(f"\nInput text:\n{sample_text}\n")
print("="*60)

# Extract entities
entities = extract_call_center_entities(sample_text)

print("\nExtracted Entities:")
print("="*60)
for key, value in entities.items():
    print(f"{key:20s}: {value}")
print("="*60)

# Test with your actual transcription if you have it
print("\n\nTo test with your actual audio transcription:")
print("1. Run your API with an audio file")
print("2. Copy the transcription text")
print("3. Replace 'sample_text' in this script")
print("4. Run: python test_extraction.py")
