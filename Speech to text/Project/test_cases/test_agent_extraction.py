#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test agent name extraction with the provided sample text"""

from modules.speech_to_text.extract import EntityExtractor

# User's sample text
sample_text = """مساء النور يا فندم، شكراً على اتصالك مع حضرتك ياسمين، ممكن أعرف اسم حضرتك؟ مساء النور يا أستاذة ياسمين، مع حضرتك محمود سامي. أهلاً يا أستاذ محمود، اتفضل، أقدر أساعد حضرتك في إيه؟ الحقيقة يا فندم، الفاتورة الشهر ده عالية جداً عن المعتاد، وأنا مش فاهم السبب خالص. تمام يا فندم، قيمة الفاتورة وصلت لكام؟ وصلت لـ 520 جنيه، مع أني عمري ما بدفع أكتر من 300 جنيه. تمام يا فندم، لحظة هراجع البيانات. ممكن أعرف رقم الموبايل المسجل على الحساب من فضلك؟ أه، طبعاً يا فندم، الرقم هو 01012345678."""

extractor = EntityExtractor()

# Extract entities
print("=" * 60)
print("Testing Entity Extraction")
print("=" * 60)
print("\nOriginal Text:")
print(sample_text[:150] + "...")
print("\n" + "=" * 60)

# Extract all entities
entities = extractor.extract_entities(sample_text)

print("\n📋 Extracted Entities:")
print("-" * 60)
for key, value in entities.items():
    print(f"  {key:20} : {value}")
print("-" * 60)

# Specific checks
print("\n🔍 Verification:")
print(f"  ✓ Agent Name Found    : {'ياسمين' in str(entities['agent_name']) if entities['agent_name'] else '❌ NOT FOUND'}")
print(f"  ✓ Client Name Found   : {'محمود سامي' in str(entities['client_name']) if entities['client_name'] else '❌ NOT FOUND'}")
print(f"  ✓ Phone Number Found  : {'01012345678' in str(entities['phone_number']) if entities['phone_number'] else '❌ NOT FOUND'}")

print("\n" + "=" * 60)
print("✅ Test Complete!")
print("=" * 60)
