"""
💰 Billing Routing Classifier - مصنف توجيه شكاوى الفواتير
يصنف شكوى الفواتير إلى الفريق المناسب باستخدام Groq LLM

الفرق المتاحة:
- PaymentSupport: مشاكل عملية الدفع نفسها
- BillingSupport: مشاكل قيم الفواتير والحسابات
- AccountManagement: حالة الخدمة المرتبطة بالفواتير
"""
from typing import Dict, Any, Optional, Tuple
from groq import Groq
import os
import json
from .config import LLM_MODEL, GROQ_API_KEY
from prompt_loader import load_prompt
class BillingRoutingClassifier:
    """
    مصنف توجيه شكاوى الفواتير
    يحدد الفريق المسؤول عن حل مشكلة العميل
    """
    
    # معلومات الفرق
    TEAM_INFO = {
        "PaymentSupport": {
            "team": "PaymentSupport",
            "team_ar": "فريق دعم المدفوعات",
            "action": "مراجعة حالة الدفع والمعاملات المالية",
            "action_en": "Review payment status and financial transactions",
            "color": "team-payment"
        },
        "BillingSupport": {
            "team": "BillingSupport",
            "team_ar": "فريق دعم الفواتير",
            "action": "مراجعة تفاصيل الفاتورة والرسوم",
            "action_en": "Review invoice details and charges",
            "color": "team-billing"
        },
        "AccountManagement": {
            "team": "AccountManagement",
            "team_ar": "فريق إدارة الحسابات",
            "action": "مراجعة حالة الحساب وإعادة تفعيل الخدمة",
            "action_en": "Review account status and service reactivation",
            "color": "team-account"
        }
    }
    SYSTEM_PROMPT = load_prompt("billing_router_system")
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        self.client = Groq(api_key=api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY"))
        self.model = model or LLM_MODEL
    
    def classify(self, complaint_text: str) -> Dict[str, Any]:
        """
        تصنيف شكوى الفواتير إلى الفريق المناسب
        
        Args:
            complaint_text: نص شكوى العميل
            
        Returns:
            dict with team routing info
        """
        target_team = self._call_llm(complaint_text)
        
        # التحقق من صحة الفريق
        if target_team not in self.TEAM_INFO:
            target_team = self._fallback_classify(complaint_text)
        
        return self.TEAM_INFO[target_team]
    
    def _call_llm(self, complaint_text: str) -> str:
        """استدعاء LLM لتصنيف الشكوى"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": complaint_text}
                ],
                temperature=0,
                max_tokens=50,
                seed=42
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # استخراج الفريق من JSON
            return self._parse_team(result_text)
            
        except Exception as e:
            print(f"⚠️ Billing routing LLM error: {e}")
            return self._fallback_classify(complaint_text)
    
    def _parse_team(self, llm_output: str) -> str:
        """استخراج اسم الفريق من مخرجات LLM"""
        
        # محاولة parse كـ JSON
        try:
            data = json.loads(llm_output)
            team = data.get("target_team", "").strip()
            if team in self.TEAM_INFO:
                return team
        except json.JSONDecodeError:
            pass
        
        # Fallback: ابحث عن اسم الفريق في النص
        for team_name in self.TEAM_INFO:
            if team_name.lower() in llm_output.lower():
                return team_name
        
        return "BillingSupport"  # افتراضي
    
    def _fallback_classify(self, text: str) -> str:
        """تصنيف احتياطي بالكلمات المفتاحية"""
        
        payment_keywords = [
            "دفعت", "دفع", "سددت", "سداد", "اتخصم", "خصم", "خصموا",
            "سحب", "سحبوا", "اتسحب", "مرتين", "كرتين", "فلوس",
            "فيزا", "كارت", "بطاقة", "فوري", "أمان", "تحويل",
            "محفظة", "فودافون كاش", "اورنج كاش", "مدفوعات"
        ]
        
        account_keywords = [
            "فصل", "فاصل", "مقطوع", "موقوف", "مش شغال", "مقفول",
            "حسابي", "الخدمة واقفة", "واقفة", "التفعيل", "إعادة تفعيل",
            "محظور", "معلق", "مسكر"
        ]
        
        # billing_keywords (default) - فاتورة عالية، رسوم، مبلغ
        
        text_lower = text.lower()
        
        payment_score = sum(1 for kw in payment_keywords if kw in text_lower)
        account_score = sum(1 for kw in account_keywords if kw in text_lower)
        
        if payment_score > account_score and payment_score > 0:
            return "PaymentSupport"
        elif account_score > payment_score and account_score > 0:
            return "AccountManagement"
        else:
            return "BillingSupport"


# Singleton
_billing_router_instance = None

def get_billing_router(api_key: Optional[str] = None) -> BillingRoutingClassifier:
    """الحصول على instance وحيدة"""
    global _billing_router_instance
    if _billing_router_instance is None:
        _billing_router_instance = BillingRoutingClassifier(api_key=api_key)
    return _billing_router_instance
