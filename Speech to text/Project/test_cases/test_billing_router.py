"""
🧪 Test Cases for Billing Routing Classifier
اختبار مصنف توجيه شكاوى الفواتير
"""
from Complaints.billing_router import BillingRoutingClassifier


def run_tests():
    router = BillingRoutingClassifier()
    
    test_cases = [
        # ═══ PaymentSupport ═══
        {
            "complaint": "حاولت أدفع أونلاين والعملية بترفض",
            "expected_team": "PaymentSupport",
            "description": "فشل الدفع الإلكتروني"
        },
        {
            "complaint": "دفعت الفاتورة مرتين واتخصم مني المبلغ كرتين",
            "expected_team": "PaymentSupport",
            "description": "خصم مكرر"
        },
        {
            "complaint": "اتسحب مني فلوس من الفيزا ومتسجلش عندكم",
            "expected_team": "PaymentSupport",
            "description": "سحب من البطاقة بدون تسجيل"
        },
        {
            "complaint": "دفعت من فوري من ساعتين والدفع مش ظاهر",
            "expected_team": "PaymentSupport",
            "description": "دفع عبر وسيط لم يظهر"
        },
        {
            "complaint": "المحفظة اتخصم منها والفاتورة لسه مش مدفوعة",
            "expected_team": "PaymentSupport",
            "description": "خصم من المحفظة بدون تأكيد"
        },
        
        # ═══ BillingSupport ═══
        {
            "complaint": "الفاتورة الشهر ده عالية جداً مش فاهم ليه",
            "expected_team": "BillingSupport",
            "description": "فاتورة مرتفعة بدون سبب واضح"
        },
        {
            "complaint": "في رسوم إضافية في الفاتورة مش عارف هي إيه",
            "expected_team": "BillingSupport",
            "description": "رسوم غير مفهومة"
        },
        {
            "complaint": "ليه الفاتورة 500 جنيه وأنا الباقة بتاعتي 200",
            "expected_team": "BillingSupport",
            "description": "فرق بين المتوقع والفاتورة"
        },
        {
            "complaint": "عايز أفهم تفاصيل الفاتورة بتاعتي",
            "expected_team": "BillingSupport",
            "description": "استفسار عن تفاصيل الفاتورة"
        },
        
        # ═══ AccountManagement ═══
        {
            "complaint": "دفعت الفاتورة والنت لسه فاصل",
            "expected_team": "AccountManagement",
            "description": "خدمة متوقفة رغم الدفع"
        },
        {
            "complaint": "الحساب بتاعي موقوف رغم إني سددت كل المستحقات",
            "expected_team": "AccountManagement",
            "description": "حساب موقوف بعد السداد"
        },
        {
            "complaint": "سددت ومحدش فعّل الخدمة تاني",
            "expected_team": "AccountManagement",
            "description": "تأخر إعادة التفعيل بعد السداد"
        },
    ]
    
    print("=" * 70)
    print("🧪 اختبار مصنف توجيه شكاوى الفواتير")
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
