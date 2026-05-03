"""
🌐 Internet Problem Diagnosis Flow - تدفق تشخيص مشاكل الإنترنت
LLM-based automatic diagnosis for internet issues

Flow:
1. Classify complaint
2. If "مشكلة إنترنت" → LLM analyzes complaint and determines team
3. Route to appropriate team (NOC / Level 2 / Field Support)
4. Output JSON result
"""

import os
import re
import json
from groq import Groq
from datetime import datetime
import uuid

from .config import GROQ_API_KEY, LLM_MODEL
from prompt_loader import load_prompt


class InternetDiagnosisFlow:
    """LLM-based automatic diagnosis flow for internet problems"""
    
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY or os.environ.get("GROQ_API_KEY"))
        self.model = LLM_MODEL
        
        # Routing logic
        self.routing_rules = {
            "noc": {
                "team": "NOC",
                "team_ar": "فريق الشبكات",
                "action": "فحص الخط من السنترال (Line Issue)",
                "description": "مشكلة في الخط الخارجي أو السنترال"
            },
            "level2": {
                "team": "Level 2",
                "team_ar": "الدعم الفني المتقدم",
                "action": "فحص إعدادات الراوتر أو مشكلة Configuration",
                "description": "مشكلة في إعدادات الراوتر أو Authentication"
            },
            "field_support": {
                "team": "Field Support",
                "team_ar": "الدعم الميداني",
                "action": "تسجيل زيارة فني للعميل",
                "description": "يتطلب زيارة فني للموقع"
            }
        }
    
    def classify_complaint(self, complaint_text: str) -> dict:
        """Classify the complaint using LLM"""
        
        system_prompt = load_prompt("internet_diagnosis_classify_system")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"صنف هذه الشكوى:\n\n{complaint_text}"}
                ],
                temperature=0.1,
                max_tokens=20
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            if "internet" in result:
                return {"code": "internet", "name": "مشكلة إنترنت", "confidence": 0.95}
            elif "billing" in result:
                return {"code": "billing", "name": "مشكلة فواتير", "confidence": 0.95}
            elif "service" in result:
                return {"code": "service_request", "name": "طلب خدمة", "confidence": 0.95}
            else:
                return {"code": "internet", "name": "مشكلة إنترنت", "confidence": 0.60}
                
        except Exception as e:
            print(f"❌ Classification Error: {e}")
            return {"code": "unknown", "name": "غير محدد", "confidence": 0.0}
    
    def extract_customer_info(self, complaint_text: str) -> dict:
        """Extract customer information from complaint text using LLM"""
        
        system_prompt = load_prompt("internet_diagnosis_extract_info_system")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": complaint_text}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            result = response.choices[0].message.content.strip()
            # Try to parse JSON
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                # Extract JSON from response if wrapped in other text
                json_match = re.search(r'\{[^}]+\}', result, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
                return {"customer_name": None, "phone_number": None, "is_urgent": False}
                
        except Exception as e:
            print(f"❌ Extraction Error: {e}")
            return {"customer_name": None, "phone_number": None, "is_urgent": False}
    
    def determine_routing_llm(self, complaint_text: str) -> dict:
        """Determine appropriate team using LLM analysis of complaint text"""
        
        system_prompt = load_prompt("internet_diagnosis_routing_system")

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"حلل هذه الشكوى وحدد الفريق المناسب:\n\n{complaint_text}"}
                ],
                temperature=0,
                max_tokens=20
            )
            
            result = response.choices[0].message.content.strip()
            
            # Determine which team based on LLM response
            if "NOC" in result:
                return self.routing_rules["noc"]
            elif "Level 2" in result or "Level2" in result:
                return self.routing_rules["level2"]
            elif "Field" in result or "field" in result.lower():
                return self.routing_rules["field_support"]
            else:
                # Default to Level 2
                return self.routing_rules["level2"]
                
        except Exception as e:
            print(f"❌ Routing Error: {e}")
            # Default to Level 2 on error
            return self.routing_rules["level2"]
    
    def generate_json_output(
        self,
        classification: dict,
        customer_info: dict,
        routing: dict,
        complaint_text: str = ""
    ) -> dict:
        """Generate the final JSON output"""
        
        output = {
            "تصنيف_المشكلة": classification["name"],
            "معلومات_العميل": {
                "الاسم": customer_info.get("customer_name"),
                "رقم_الهاتف": customer_info.get("phone_number"),
                "حالة_طوارئ": customer_info.get("is_urgent", False)
            },
            "التحليل": {
                "وصف_المشكلة": complaint_text[:200] + "..." if len(complaint_text) > 200 else complaint_text,
                "الفريق_المناسب": routing["team"],
                "اسم_الفريق_بالعربي": routing["team_ar"],
                "الوصف": routing["description"]
            },
            "الإجراء_المقترح": routing["action"],
            "رقم_التذكرة": f"TKT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
            "التوقيت": datetime.now().isoformat(),
            "طريقة_التوجيه": "LLM-based (Groq)"
        }
        
        return output
    
    def run_flow(self, complaint_text: str) -> dict:
        """Run the complete LLM-based diagnosis flow"""
        
        print("\n" + "🔄"*30)
        print("   نظام تشخيص مشاكل الإنترنت (LLM)")
        print("   Internet Problem Diagnosis System (LLM-based)")
        print("🔄"*30)
        
        # Step 1: Classify complaint
        print("\n" + "="*60)
        print("📌 الخطوة 1: تصنيف الشكوى")
        print("="*60)
        
        classification = self.classify_complaint(complaint_text)
        print(f"✅ التصنيف: {classification['name']}")
        print(f"   الثقة: {classification['confidence']:.0%}")
        
        # Step 2: Extract customer info
        print("\n" + "="*60)
        print("📌 الخطوة 2: استخراج معلومات العميل")
        print("="*60)
        
        customer_info = self.extract_customer_info(complaint_text)
        print(f"👤 اسم العميل: {customer_info.get('customer_name', 'غير متوفر')}")
        print(f"📞 رقم الهاتف: {customer_info.get('phone_number', 'غير متوفر')}")
        print(f"🚨 حالة طوارئ: {'نعم' if customer_info.get('is_urgent') else 'لا'}")
        
        # Step 3: Check if internet problem
        if classification["code"] != "internet":
            print(f"\n⚠️ المشكلة ليست مشكلة إنترنت - التصنيف: {classification['name']}")
            print("   يتم تحويل الشكوى للقسم المختص...")
            
            return {
                "تصنيف_المشكلة": classification["name"],
                "معلومات_العميل": customer_info,
                "ملاحظة": "ليست مشكلة إنترنت - لا يتطلب تشخيص",
                "رقم_التذكرة": f"TKT-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
            }
        
        # Step 4: LLM-based automatic routing (no questions needed)
        print("\n" + "="*60)
        print("📌 الخطوة 3: التحليل التلقائي بواسطة LLM")
        print("="*60)
        print("🤖 جاري تحليل الشكوى لتحديد الفريق المناسب...")
        
        routing = self.determine_routing_llm(complaint_text)
        
        print(f"\n✅ اكتمل التحليل:")
        print(f"   👥 الفريق: {routing['team']} - {routing['team_ar']}")
        print(f"   📋 الوصف: {routing['description']}")
        print(f"   🔧 الإجراء: {routing['action']}")
        
        # Step 5: Generate JSON output
        print("\n" + "="*60)
        print("📌 الخطوة 4: المخرجات النهائية (JSON)")
        print("="*60)
        
        result = self.generate_json_output(
            classification=classification,
            customer_info=customer_info,
            routing=routing,
            complaint_text=complaint_text
        )
        
        print(json.dumps(result, ensure_ascii=False, indent=2))
        
        return result


def run_with_sample_text():
    """Run the flow with the sample text from test_rag_full_pipeline.py"""
    
    sample_text = """
مساء الخير يا فندم شكرا لاتصالك ب WE للاتصالات احمد في خدمتك ممكن اتشرف باسم حضرتك؟ اهلا استاذ احمد انا اسمي هبة والحقيقة ان انا موجودة علي خدمة الانتظار بقالي فترة طويلة جدا انا بعتذر جدا يا فندم الوقت ده من اليوم بيكون فيه ضغط ومكالمات كبير جدا اسفين جدا علي وقت حضرتك ممكن حضرتك تبلغيني بالرقم المتاح عليه الخدمة؟ طبعا الرقم هو 01274714823 تمام يا فندم هل الخط مسجل باسم حضرتك؟ مسجل باسمي طيب ممكن توضحي لي الاسم بالكامل يا فندم؟ هبة عاطف الناغي عفيفي تمام شكرا جدا يا يا استاذة هبة اقدر اساعدك في ايه يا فندم؟ انا اساعد حضرتك ازاي؟ الحقيقة يا استاذ احمد انا عايز اقول لك ان انا بالضبط علي بعد 30 دقيقة من اني اتفصل من شغلي انا كنت بشتغل والانترنت شغال مشكلة طبيعي وبعد ما حضرت ملفات الاكسل والاميلات اللي كنت محتاج ابعتها قبل ما اضبط علي الارسال فجاة الانترنت فصل وعلامة الانترنت علي جهاز الكمبيوتر بتقول ان مفيش انترنت والموضوع مش منطقي خالص لاني دفعت الفاتورة في معادها الساعة دلوقتي 9 وانا محتاج ابعت الشغل بتاعي الساعة 10 من الكتير فمن فضلك استاذ احمد انا محتاج حالة لان الوقت لو عد من غير ما ابعت الاميلات دي هتكون مشكلة كبيرة في شغلي.
"""
    
    flow = InternetDiagnosisFlow()
    
    print("\n" + "📝"*30)
    print("   نص الشكوى الأصلي:")
    print("📝"*30)
    print(sample_text[:300] + "...")
    
    # Run LLM-based flow (automatic, no interaction needed)
    result = flow.run_flow(sample_text)
    
    return result


def run_automated_test():
    """Run automated test with different scenarios"""
    
    test_scenarios = [
        {
            "name": "انقطاع كامل - لمبة DSL حمراء",
            "text": "الإنترنت قاطع تماماً من امبارح ولمبة DSL في الراوتر حمراء. اسمي أحمد ورقمي 01012345678",
            "expected_team": "NOC"
        },
        {
            "name": "لمبة DSL مطفية",
            "text": "النت مش شغال خالص ولمبة DSL مطفية. محتاج حل عاجل. اسمي سارة",
            "expected_team": "NOC"
        },
        {
            "name": "DSL أخضر لكن مفيش نت",
            "text": "لمبة DSL خضراء لكن الإنترنت مش شغال. مش عارف المشكلة فين. اسمي محمد",
            "expected_team": "Level 2"
        },
        {
            "name": "جرب كل الحلول",
            "text": "انا جربت افصل الراوتر واعيد تشغيله اكتر من مرة والنت برضو مش شغال. لمبة DSL خضراء. محتاج فني يجي يشوف المشكلة. اسمي علي",
            "expected_team": "Field Support"
        },
        {
            "name": "انترنت بطيء",
            "text": "الإنترنت بطيء جداً من كام يوم. السرعة مش زي الأول خالص. اسمي نور",
            "expected_team": "Level 2"
        },
        {
            "name": "انقطاع متكرر",
            "text": "الانترنت بيفصل ويرجع كل شوية. الموضوع بقى مزعج جداً. اسمي هبة",
            "expected_team": "Level 2"
        }
    ]
    
    flow = InternetDiagnosisFlow()
    
    print("\n" + "🧪"*30)
    print("   اختبار السيناريوهات المختلفة")
    print("   Testing Different Scenarios")
    print("🧪"*30)
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n\n{'='*70}")
        print(f"🔬 السيناريو {i}: {scenario['name']}")
        print(f"   المتوقع: {scenario['expected_team']}")
        print("="*70)
        
        result = flow.run_flow(scenario['text'])
        actual_team = result.get('التحليل', {}).get('الفريق_المناسب', 'Unknown')
        
        # Check if routing matches expected
        if actual_team == scenario['expected_team']:
            print(f"\n   ✅ صحيح! الفريق: {actual_team}")
            status = "✅ PASS"
        else:
            print(f"\n   ⚠️ توقع: {scenario['expected_team']}, نتيجة: {actual_team}")
            status = "⚠️ REVIEW"
        
        results.append({
            "scenario": scenario['name'],
            "expected": scenario['expected_team'],
            "actual": actual_team,
            "status": status
        })
    
    # Print summary
    print("\n\n" + "="*70)
    print("📊 ملخص النتائج - Results Summary")
    print("="*70)
    
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['scenario']}")
        print(f"   متوقع: {res['expected']} | نتيجة: {res['actual']} | {res['status']}")
    
    passed = sum(1 for r in results if "PASS" in r['status'])
    print(f"\n🎯 النتيجة النهائية: {passed}/{len(results)} اختبار ناجح")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run automated tests
        run_automated_test()
    else:
        # Run with sample text
        run_with_sample_text()
