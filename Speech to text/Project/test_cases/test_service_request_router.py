"""
🧪 Test Cases for Service Request Routing Classifier
اختبار مصنف توجيه طلبات الخدمة
"""
from Complaints.service_request_router import ServiceRequestRoutingClassifier


def run_tests():
    router = ServiceRequestRoutingClassifier()
    
    test_cases = [
        # ═══ PlanUpgradeManagement ═══
        {
            "complaint": "عايز أغير الباقة لباقة أسرع",
            "expected_team": "PlanUpgradeManagement",
            "description": "ترقية الباقة"
        },
        {
            "complaint": "محتاج أنزل لباقة أرخص الشهر ده",
            "expected_team": "PlanUpgradeManagement",
            "description": "تخفيض الباقة"
        },
        {
            "complaint": "عايز أضيف خدمة الـ Static IP على الخط بتاعي",
            "expected_team": "PlanUpgradeManagement",
            "description": "إضافة خدمة على اشتراك قائم"
        },
        {
            "complaint": "في عرض أحسن ليا على الباقة الحالية؟",
            "expected_team": "PlanUpgradeManagement",
            "description": "استفسار عن عروض للباقة الحالية"
        },
        {
            "complaint": "عايز ألغي خدمة مقوي الشبكة من الاشتراك",
            "expected_team": "PlanUpgradeManagement",
            "description": "إلغاء خدمة إضافية"
        },
        
        # ═══ ServiceProvisioning ═══
        {
            "complaint": "عايز أركب خط إنترنت جديد في البيت",
            "expected_team": "ServiceProvisioning",
            "description": "تركيب خط جديد"
        },
        {
            "complaint": "سجلت اشتراك جديد من أسبوع ولسه مجاش حد يركب",
            "expected_team": "ServiceProvisioning",
            "description": "تأخر تركيب اشتراك جديد"
        },
        {
            "complaint": "الخدمة متفعلتش لسه بعد التركيب",
            "expected_team": "ServiceProvisioning",
            "description": "عدم تفعيل بعد التركيب"
        },
        {
            "complaint": "عايز أعرف إيه المستندات المطلوبة لاشتراك جديد",
            "expected_team": "ServiceProvisioning",
            "description": "استفسار عن اشتراك جديد"
        },
        
        # ═══ ServiceQualitySupport ═══
        {
            "complaint": "النت بطيء جداً من كام يوم",
            "expected_team": "ServiceQualitySupport",
            "description": "بطء الإنترنت"
        },
        {
            "complaint": "الخدمة بتقطع كل شوية ومش مستقرة",
            "expected_team": "ServiceQualitySupport",
            "description": "انقطاع متكرر"
        },
        {
            "complaint": "جودة الاتصال وحشة والسرعة أقل من المفروض",
            "expected_team": "ServiceQualitySupport",
            "description": "جودة خدمة سيئة"
        },
        {
            "complaint": "الواي فاي ضعيف ومش واصل للغرف البعيدة",
            "expected_team": "ServiceQualitySupport",
            "description": "ضعف إشارة الواي فاي"
        },
    ]
    
    print("=" * 70)
    print("🧪 اختبار مصنف توجيه طلبات الخدمة")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for i, tc in enumerate(test_cases, 1):
        result = router.classify(tc["complaint"])
        actual_team = result["team"]
        status = "✅" if actual_team == tc["expected_team"] else "❌"
        
        if actual_team == tc["expected_team"]:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} Test {i}: {tc['description']}")
        print(f"   الشكوى: {tc['complaint']}")
        print(f"   المتوقع: {tc['expected_team']} | الفعلي: {actual_team} ({result['team_ar']})")
    
    print("\n" + "=" * 70)
    print(f"📊 النتائج: {passed} نجح | {failed} فشل | الإجمالي {len(test_cases)}")
    print(f"   نسبة النجاح: {passed/len(test_cases)*100:.0f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
