"""
🛠️ Service Request Routing Classifier - مصنف توجيه طلبات الخدمة
يصنف طلب الخدمة إلى الفريق المناسب باستخدام Groq LLM

الفرق المتاحة:
- PlanUpgradeManagement: تغيير أو تعديل خدمة موجودة
- ServiceProvisioning: خدمة جديدة أو تفعيل
- ServiceQualitySupport: جودة الخدمة ومشاكل الأداء
"""

from typing import Dict, Any, Optional
from groq import Groq
import os
import json

from .config import LLM_MODEL, GROQ_API_KEY
from prompt_loader import load_prompt


class ServiceRequestRoutingClassifier:
    """
    مصنف توجيه طلبات الخدمة
    يحدد الفريق المسؤول عن تنفيذ طلب العميل
    """
    
    TEAM_INFO = {
        "PlanUpgradeManagement": {
            "team": "PlanUpgradeManagement",
            "team_ar": "فريق إدارة ترقية الباقات",
            "action": "مراجعة الباقة الحالية وتنفيذ التعديل المطلوب",
            "action_en": "Review current plan and execute requested modification",
            "color": "team-upgrade"
        },
        "ServiceProvisioning": {
            "team": "ServiceProvisioning",
            "team_ar": "فريق توفير الخدمات",
            "action": "تجهيز وتفعيل الخدمة الجديدة",
            "action_en": "Provision and activate new service",
            "color": "team-provision"
        },
        "ServiceQualitySupport": {
            "team": "ServiceQualitySupport",
            "team_ar": "فريق دعم جودة الخدمة",
            "action": "فحص جودة الخدمة وحل مشاكل الأداء",
            "action_en": "Inspect service quality and resolve performance issues",
            "color": "team-quality"
        }
    }
    
    SYSTEM_PROMPT = load_prompt("service_request_router_system")
    
    def __init__(self, api_key: Optional[str] = None, model: str = None):
        self.client = Groq(api_key=api_key or GROQ_API_KEY or os.getenv("GROQ_API_KEY"))
        self.model = model or LLM_MODEL
    
    def classify(self, complaint_text: str) -> Dict[str, Any]:
        """
        تصنيف طلب الخدمة إلى الفريق المناسب
        
        Args:
            complaint_text: نص طلب العميل
            
        Returns:
            dict with team routing info
        """
        target_team = self._call_llm(complaint_text)
        
        if target_team not in self.TEAM_INFO:
            target_team = self._fallback_classify(complaint_text)
        
        return self.TEAM_INFO[target_team]
    
    def _call_llm(self, complaint_text: str) -> str:
        """استدعاء LLM لتصنيف الطلب"""
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
            return self._parse_team(result_text)
            
        except Exception as e:
            print(f"⚠️ Service request routing LLM error: {e}")
            return self._fallback_classify(complaint_text)
    
    def _parse_team(self, llm_output: str) -> str:
        """استخراج اسم الفريق من مخرجات LLM"""
        
        try:
            data = json.loads(llm_output)
            team = data.get("target_team", "").strip()
            if team in self.TEAM_INFO:
                return team
        except json.JSONDecodeError:
            pass
        
        for team_name in self.TEAM_INFO:
            if team_name.lower() in llm_output.lower():
                return team_name
        
        return "ServiceQualitySupport"
    
    def _fallback_classify(self, text: str) -> str:
        """تصنيف احتياطي بالكلمات المفتاحية"""
        
        upgrade_keywords = [
            "أغير", "تغيير", "غير", "باقة", "ترقية", "رقي", "أرقي",
            "خفض", "أنزل", "عايز باقة", "سرعة أعلى", "سرعة أكبر",
            "عرض أحسن", "باقة أحسن", "باقة أرخص", "أضيف", "ألغي خدمة",
            "أشيل", "أضيف خدمة", "تعديل", "عدّل"
        ]
        
        provision_keywords = [
            "تركيب", "ركب", "خط جديد", "اشتراك جديد", "اشتراك",
            "تفعيل", "فعّل", "متفعلش", "مش متفعل", "جديد",
            "أول مرة", "تأسيس", "توصيل", "ركبوا", "مستني التركيب"
        ]
        
        quality_keywords = [
            "بطيء", "بطيئ", "ضعيف", "ضعيفة", "واقع", "بيقطع",
            "مش شغال كويس", "وحش", "مشاكل", "بيفصل", "انقطاع",
            "سرعة قليلة", "الجودة", "أداء"
        ]
        
        text_lower = text.lower()
        
        upgrade_score = sum(1 for kw in upgrade_keywords if kw in text_lower)
        provision_score = sum(1 for kw in provision_keywords if kw in text_lower)
        quality_score = sum(1 for kw in quality_keywords if kw in text_lower)
        
        scores = {
            "PlanUpgradeManagement": upgrade_score,
            "ServiceProvisioning": provision_score,
            "ServiceQualitySupport": quality_score
        }
        
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
        return "ServiceQualitySupport"


# Singleton
_service_router_instance = None

def get_service_request_router(api_key: Optional[str] = None) -> ServiceRequestRoutingClassifier:
    """الحصول على instance وحيدة"""
    global _service_router_instance
    if _service_router_instance is None:
        _service_router_instance = ServiceRequestRoutingClassifier(api_key=api_key)
    return _service_router_instance
