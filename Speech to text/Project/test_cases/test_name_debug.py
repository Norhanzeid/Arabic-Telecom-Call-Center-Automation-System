import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(__file__))

from modules.speech_to_text.extract import EntityExtractor, extract_call_center_entities

extractor = EntityExtractor()

test_cases = [
    "معك سعدي محمد القحطاني ورقم جوالي 0551234567",
    "معاك سعدي محمد القحطاني",
    "أنا اسمي سعدي محمد القحطاني",
    "اسمي سعدي محمد القحطاني",
    "معك محمد عبدالله الشمري",
    "معاك فهد سعود العتيبي ومشكلتي ان النت فصل",
    "مع سعدي محمد القحطاني",
    "معاكم سعدي محمد القحطاني",
    "السلام عليكم معك سعدي محمد القحطاني عندي مشكلة في النت",
]

print("=" * 70)
print("TEST: extract_client_name")
print("=" * 70)
for text in test_cases:
    name = extractor.extract_client_name(text)
    status = "PASS" if name else "FAIL"
    print(f"[{status}] Input:  {text}")
    print(f"        Result: {name}")
    print("-" * 70)

print("\nTEST: Full extract_call_center_entities")
print("=" * 70)
test_full = "السلام عليكم معك سعدي محمد القحطاني ورقم جوالي 0551234567 عندي مشكلة في النت فصل من امبارح"
entities = extract_call_center_entities(test_full)
for k, v in entities.items():
    print(f"  {k}: {v}")
