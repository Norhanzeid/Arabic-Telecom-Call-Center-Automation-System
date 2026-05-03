#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test the /process-full endpoint with sample text
"""

import requests
import json

# API endpoint
API_URL = "http://localhost:8000"

# Sample text (billing complaint)
sample_text = """مساء النور يا فندم، شكراً على اتصالك مع حضرتك ياسمين، ممكن أعرف اسم حضرتك؟ مساء النور يا أستاذة ياسمين، مع حضرتك محمود سامي. أهلاً يا أستاذ محمود، اتفضل، أقدر أساعد حضرتك في إيه؟ الحقيقة يا فندم، الفاتورة الشهر ده عالية جداً عن المعتاد، وأنا مش فاهم السبب خالص. تمام يا فندم، قيمة الفاتورة وصلت لكام؟ وصلت لـ 520 جنيه، مع أني عمري ما بدفع أكتر من 300 جنيه. تمام يا فندم، لحظة هراجع البيانات. ممكن أعرف رقم الموبايل المسجل على الحساب من فضلك؟ أه، طبعاً يا فندم، الرقم هو 01012345678."""


def test_full_pipeline_with_text():
    """Test the complete pipeline with text input"""
    print("=" * 70)
    print("Testing /process-full endpoint with TEXT")
    print("=" * 70)
    
    # Prepare request
    params = {
        "text": sample_text,
        "extract_entities": True,
        "diarize": True
    }
    
    print("\n📤 Sending request to API...")
    try:
        response = requests.post(
            f"{API_URL}/process-full",
            params=params,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ SUCCESS!")
            print("=" * 70)
            
            # Print structured results
            print(f"\n🆔 Processing ID: {result.get('processing_id')}")
            print(f"⏰ Timestamp: {result.get('timestamp')}")
            
            print("\n📝 Classification:")
            classification = result.get('classification', {})
            print(f"   Category: {classification.get('category_code')} - {classification.get('category_name')}")
            print(f"   Confidence: {classification.get('confidence'):.0%}")
            
            if 'entities' in result:
                print("\n👤 Entities Extracted:")
                entities = result['entities']
                print(f"   Client Name: {entities.get('client_name', 'N/A')}")
                print(f"   Phone Number: {entities.get('phone_number', 'N/A')}")
                print(f"   Agent Name: {entities.get('agent_name', 'N/A')}")
            
            if 'diarization' in result:
                print(f"\n💬 Diarization: {result['diarization']['turns_count']} turns")
            
            if 'team_routing' in result:
                print("\n🎯 Team Routing:")
                routing = result['team_routing']
                print(f"   Team: {routing.get('team')} - {routing.get('team_ar')}")
                print(f"   Action: {routing.get('action')}")
            
            if 'internet_diagnosis' in result:
                print("\n🔧 Internet Diagnosis:")
                diagnosis = result['internet_diagnosis'].get('التحليل', {})
                print(f"   Team: {diagnosis.get('الفريق_المناسب')} - {diagnosis.get('اسم_الفريق_بالعربي')}")
                print(f"   Action: {diagnosis.get('الإجراء_المقترح')}")
            
            print("\n" + "=" * 70)
            print("📄 Full JSON Response:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error!")
        print("Make sure the API is running:")
        print("   cd Project")
        print("   uvicorn Complaints.api:app --reload --port 8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def test_health_check():
    """Test the /health endpoint"""
    print("\n" + "=" * 70)
    print("Testing /health endpoint")
    print("=" * 70)
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy!")
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API!")
        print("Start the API with: uvicorn Complaints.api:app --reload --port 8000")


if __name__ == "__main__":
    print("\n🚀 Testing Complete API Pipeline")
    print("API URL:", API_URL)
    
    # Test health first
    test_health_check()
    
    # Test full pipeline
    test_full_pipeline_with_text()
    
    print("\n✅ Test Complete!")

