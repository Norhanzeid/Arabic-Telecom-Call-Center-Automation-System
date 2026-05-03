"""
🏷️ Complaint Classifier - مصنف الشكاوى
يصنف شكوى العميل إلى فئة واحدة فقط

الفئات:
- مشكلة نت (internet)
- مشكلة فواتير (billing)  
- مشكلة طلب خدمة (service_request)
"""
from typing import Tuple, Optional
from openai import OpenAI
import os
from prompt_loader import load_prompt


class ComplaintClassifier:
    """
    مصنف الشكاوى باستخدام LLM
    يصنف كل شكوى إلى فئة واحدة فقط
    """
    
    # كلمات مفتاحية لكل فئة (للتصنيف السريع)
    INTERNET_KEYWORDS = [
        "نت", "انترنت", "إنترنت", "النت", "الانترنت", "الإنترنت",
        "فصل", "قطع", "بطيء", "بطيئ", "ضعيف", "مش شغال", "مش شغاله",
        "واي فاي", "wifi", "واي-فاي", "راوتر", "الراوتر",
        "سرعة", "السرعة", "الاتصال", "اتصال", "متصل", "disconnected",
        "انقطاع", "انقطع", "فاصل", "واقف", "وقف"
    ]
    
    BILLING_KEYWORDS = [
        "فاتورة", "الفاتورة", "فواتير", "دفع", "دفعت", "سداد", "سددت",
        "مبلغ", "المبلغ", "فلوس", "الفلوس", "رسوم", "الرسوم",
        "خصم", "اتخصم", "خصموا", "سحب", "سحبوا", "اتسحب",
        "متأخر", "مستحق", "المستحق", "رصيد", "الرصيد",
        "عالية", "مرتفعة", "غالية", "كتير", "زيادة", "مرتين"
    ]
    
    SERVICE_KEYWORDS = [
        "باقة", "الباقة", "تغيير", "غير", "أغير", "عايز أغير",
        "راوتر جديد", "راوتر تاني", "تركيب", "ركب", "أركب",
        "خط جديد", "اشتراك جديد", "اشتراك", "إلغاء", "الغي", "ألغي",
        "نقل", "انقل", "تنقيل", "ملكية", "باسم",
        "ترقية", "رقي", "خفض", "خدمة", "خدمات", "طلب"
    ]
    
    CATEGORIES = {
        "internet": "مشكلة نت",
        "billing": "مشكلة فواتير",
        "service_request": "مشكلة طلب خدمة"
    }
    
    def __init__(self, use_llm: bool = True, api_key: Optional[str] = None):
        """
        تهيئة المصنف
        
        Args:
            use_llm: استخدام LLM للتصنيف (أدق) أو الكلمات المفتاحية (أسرع)
            api_key: مفتاح OpenAI API
        """
        self.use_llm = use_llm
        if use_llm:
            self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def classify(self, complaint_text: str) -> Tuple[str, str, float]:
        """
        تصنيف الشكوى إلى فئة واحدة
        
        Args:
            complaint_text: نص الشكوى
            
        Returns:
            Tuple of (category_code, category_name_arabic, confidence)
        """
        if self.use_llm:
            return self._classify_with_llm(complaint_text)
        else:
            return self._classify_with_keywords(complaint_text)
    
    def _classify_with_keywords(self, text: str) -> Tuple[str, str, float]:
        """
        تصنيف سريع باستخدام الكلمات المفتاحية
        """
        text_lower = text.lower()
        
        # حساب عدد التطابقات لكل فئة
        internet_score = sum(1 for kw in self.INTERNET_KEYWORDS if kw in text_lower)
        billing_score = sum(1 for kw in self.BILLING_KEYWORDS if kw in text_lower)
        service_score = sum(1 for kw in self.SERVICE_KEYWORDS if kw in text_lower)
        
        # تحديد الفئة بأعلى نتيجة
        scores = {
            "internet": internet_score,
            "billing": billing_score,
            "service_request": service_score
        }
        
        max_category = max(scores, key=scores.get)
        total = sum(scores.values()) or 1
        confidence = scores[max_category] / total
        
        return max_category, self.CATEGORIES[max_category], confidence
    
    def _classify_with_llm(self, text: str) -> Tuple[str, str, float]:
        """
        تصنيف دقيق باستخدام LLM
        """
        system_prompt = load_prompt("classifier_system")

        user_prompt = f"صنف الشكوى التالية:\n\n{text}"
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=50
            )
            
            result = response.choices[0].message.content
            
            # استخراج الفئة والثقة من الرد
            category = "internet"  # افتراضي
            confidence = 0.8
            
            for line in result.split("\n"):
                if "CATEGORY:" in line:
                    cat = line.split(":")[-1].strip().lower()
                    if cat in self.CATEGORIES:
                        category = cat
                elif "CONFIDENCE:" in line:
                    try:
                        confidence = float(line.split(":")[-1].strip())
                    except:
                        pass
            
            return category, self.CATEGORIES[category], confidence
            
        except Exception as e:
            print(f"Error in LLM classification: {e}")
            # Fallback to keyword classification
            return self._classify_with_keywords(text)


# للاستخدام المباشر
def classify_complaint(text: str, use_llm: bool = False) -> dict:
    """
    دالة مساعدة لتصنيف الشكوى
    
    Args:
        text: نص الشكوى
        use_llm: استخدام LLM أو الكلمات المفتاحية
        
    Returns:
        dict with category info
    """
    classifier = ComplaintClassifier(use_llm=use_llm)
    code, name, confidence = classifier.classify(text)
    
    return {
        "category_code": code,
        "category_name": name,
        "confidence": confidence,
        "complaint_text": text
    }


# أمثلة للاختبار
if __name__ == "__main__":
    test_complaints = [
        "النت فصل فجأة",
        "دفعت الفاتورة والإنترنت مش شغال",
        "النت قطع وأنا محتاجة أبعث شغل مهم",
        "دفعت ولسه مكتوب عليا متأخر",
        "الفاتورة عالية جدًا",
        "اتخصم مني مرتين",
        "عايز أغير الباقة",
        "محتاج راوتر جديد",
        "عايز ألغي الخدمة"
    ]
    
    classifier = ComplaintClassifier(use_llm=False)
    
    print("=" * 60)
    print("اختبار تصنيف الشكاوى")
    print("=" * 60)
    
    for complaint in test_complaints:
        code, name, confidence = classifier.classify(complaint)
        print(f"\nالشكوى: {complaint}")
        print(f"التصنيف: {name} ({code})")
        print(f"الثقة: {confidence:.2%}")
