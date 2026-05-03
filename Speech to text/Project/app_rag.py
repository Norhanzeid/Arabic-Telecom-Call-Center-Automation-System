"""
Injaz Tech - AI Platform
Speech to Text + Feature Extraction + LLM Response Generation
Uses Groq LLM (Llama 3.3) for Classification and Response Generation
"""

import streamlit as st
import os
import sys
import tempfile
from datetime import datetime

# Load environment variables FIRST
from dotenv import load_dotenv
load_dotenv()

# Set Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

# Add Project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
from modules.speech_to_text.task import run
from modules.speech_to_text.extract import extract_call_center_entities
from modules.speech_to_text.openai_whisper import transcribe_with_timestamps
from modules.speech_to_text.arabic_corrector import correct_arabic_text_gpt, apply_word_corrections

# Import LLM-based diarization
from LLM_turns import diarize_with_llm

# Import Routers & Diagnosis Flow
from Complaints.billing_router import get_billing_router
from Complaints.service_request_router import get_service_request_router
from Complaints.internet_diagnosis_flow import InternetDiagnosisFlow

# Import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq not installed. Run: pip install groq")


# =============================================================================
# Groq LLM Classes
# =============================================================================
class GroqClassifier:
    """LLM-based Arabic complaint classifier using Groq"""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model
        self.system_prompt = """أنت مصنف شكاوى لشركة اتصالات WE في مصر. مهمتك تحديد المشكلة الأساسية للعميل.

الفئات الثلاثة:
1. **internet** - مشكلة نت: انقطاع الإنترنت، النت فصل، مفيش نت، بطء السرعة، مشاكل الراوتر، الواي فاي لا يعمل، الإنترنت مش شغال، الخدمة بطيئة، جودة الخدمة سيئة
2. **billing** - مشكلة فواتير: فاتورة غلط، فاتورة عالية، خصم مش صح، عايز أعرف الفاتورة، خصموا مرتين، فلوس اتخصمت غلط، مشكلة في الدفع
3. **service_request** - طلب خدمة: تغيير الباقة، طلب راوتر جديد، نقل الخط، إلغاء الاشتراك، تغيير السرعة، ترقية الباقة، خدمة جديدة

⚠️ قواعد مهمة جداً للتصنيف:
1. 🔴 لو العميل قال أي من: "الانترنت فصل", "النت فصل", "مفيش نت", "مفيش انترنت", "بطيء", "جودة سيئة" → المشكلة **internet**
2. 🔴 لو العميل قال "دفعت الفاتورة" كدليل على الدفع لكن النت فاصل → المشكلة **internet** (الفاتورة مش المشكلة، النت هو المشكلة)
3. 🔴 لو العميل "محتاج" أو "عايز مساعدة" بس المشكلة الفعلية هي نت فاصل أو بطيء → المشكلة **internet**
4. 🔴 service_request فقط لو العميل عايز يغير باقته أو يطلب خدمة جديدة - مش لو عنده مشكلة
5. ⚠️ ركز على السبب الأساسي للاتصال مش على الكلمات الفردية

🎯 أجب بكلمة واحدة فقط: internet أو billing أو service_request"""

    def classify(self, complaint_text: str) -> tuple:
        """Classify complaint using Groq LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"صنف هذه الشكوى:\n\n{complaint_text}"}
                ],
                temperature=0,  # 0 = deterministic (same answer every time)
                max_tokens=20
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            if "internet" in result:
                return ("internet", "مشكلة نت", 0.95)
            elif "billing" in result:
                return ("billing", "مشكلة فواتير", 0.95)
            elif "service" in result:
                return ("service_request", "طلب خدمة", 0.95)
            else:
                return ("internet", "مشكلة نت", 0.60)
                
        except Exception as e:
            print(f"❌ LLM Classification Error: {e}")
            return ("internet", "مشكلة نت", 0.50)


def merge_consecutive_speaker_messages(diarized_conversation: list) -> list:
    """
    Merge consecutive messages from the same speaker into one message.
    This makes the conversation cleaner when a speaker has multiple consecutive turns.
    """
    if not diarized_conversation:
        return []
    
    merged = []
    current_speaker = None
    current_text = ""
    
    for turn in diarized_conversation:
        speaker = turn.get('speaker', 'Unknown')
        text = turn.get('text', '').strip()
        
        if speaker == current_speaker:
            # Same speaker, append text
            current_text += " " + text
        else:
            # Different speaker, save previous and start new
            if current_speaker is not None and current_text:
                merged.append({
                    'speaker': current_speaker,
                    'text': current_text.strip()
                })
            current_speaker = speaker
            current_text = text
    
    # Don't forget the last message
    if current_speaker is not None and current_text:
        merged.append({
            'speaker': current_speaker,
            'text': current_text.strip()
        })
    
    return merged





# =============================================================================
# Page Configuration
# =============================================================================
st.set_page_config(
    page_title="Injaz Tech | AI Platform",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# Custom CSS Styling
# =============================================================================
st.markdown("""
<style>
    /* ═══════════════════════════════════════════════════════════════════
       🎨 Color Palette - متناسقة مع اللوجو
       Primary: #1A81A5 (Blue)
       Secondary: #5AA7C0 (Light Blue)
       Background: #f8fafb
       Text: #212529
    ═══════════════════════════════════════════════════════════════════ */
    
    :root {
        --primary-color: #1A81A5;
        --primary-light: #5AA7C0;
        --bg-light: #f8fafb;
        --success-color: #10B981;
        --warning-color: #F59E0B;
        --danger-color: #EF4444;
        --text-dark: #212529;
        --text-muted: #6B7280;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       🏠 Main Background
    ═══════════════════════════════════════════════════════════════════ */
    .stApp {
        background-color: #f8fafb;
    }
    
    .block-container {
        background-color: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        margin-top: 1rem;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       📌 Header Styling
    ═══════════════════════════════════════════════════════════════════ */
    .main-header {
        background: linear-gradient(135deg, #1A81A5 0%, #5AA7C0 100%);
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 8px 24px rgba(26, 129, 165, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 26px;
        margin: 0;
        font-weight: 600;
    }
    
    .main-header p {
        color: rgba(255,255,255,0.95);
        font-size: 14px;
        margin-top: 8px;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       📝 Section Headers
    ═══════════════════════════════════════════════════════════════════ */
    .section-header {
        background: linear-gradient(90deg, #1A81A5, #5AA7C0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 22px;
        font-weight: 700;
        margin: 25px 0 15px 0;
        display: inline-block;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       🔘 Button Styling
    ═══════════════════════════════════════════════════════════════════ */
    .stButton > button {
        background: linear-gradient(135deg, #1A81A5 0%, #5AA7C0 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(26, 129, 165, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(26, 129, 165, 0.5);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       📂 File Uploader
    ═══════════════════════════════════════════════════════════════════ */
    .stFileUploader > div {
        box-shadow: 0 2px 12px rgba(26, 129, 165, 0.15);
        border-radius: 10px;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       ➖ Custom Divider
    ═══════════════════════════════════════════════════════════════════ */
    .custom-divider {
        height: 3px;
        background: linear-gradient(90deg, #1A81A5, #5AA7C0, #1A81A5);
        border: none;
        border-radius: 2px;
        margin: 20px 0;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       🏷️ Entity Card
    ═══════════════════════════════════════════════════════════════════ */
    .entity-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 4px solid #1A81A5;
        padding: 15px 20px;
        margin: 10px 0;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .entity-card:hover {
        transform: translateX(5px);
        box-shadow: 0 4px 15px rgba(26, 129, 165, 0.15);
    }
    
    .entity-label {
        color: #1A81A5;
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 5px;
    }
    
    .entity-value {
        color: #333;
        font-size: 16px;
        font-weight: 600;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       🤖 RAG Response Box
    ═══════════════════════════════════════════════════════════════════ */
    .rag-response-box {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border-left: 5px solid #10B981;
        padding: 25px;
        margin: 20px 0;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15);
    }
    
    .rag-response-title {
        color: #065F46;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .rag-response-text {
        color: #1F2937;
        font-size: 16px;
        line-height: 1.9;
        white-space: pre-line;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       🏷️ Classification Badges
    ═══════════════════════════════════════════════════════════════════ */
    .classification-badge {
        display: inline-block;
        padding: 10px 25px;
        border-radius: 30px;
        font-size: 15px;
        font-weight: 700;
        margin: 5px;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
    }
    
    .badge-internet {
        background: linear-gradient(135deg, #1A81A5, #5AA7C0);
        color: white;
    }
    
    .badge-billing {
        background: linear-gradient(135deg, #D97706, #F59E0B);
        color: white;
    }
    
    .badge-service {
        background: linear-gradient(135deg, #059669, #10B981);
        color: white;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       📋 Pipeline Step
    ═══════════════════════════════════════════════════════════════════ */
    .pipeline-step {
        background: white;
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        padding: 16px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .pipeline-step.active {
        border-color: #5AA7C0;
        box-shadow: 0 4px 15px rgba(26, 129, 165, 0.2);
    }
    
    .pipeline-step.completed {
        border-color: #10B981;
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       ✅ Success Box
    ═══════════════════════════════════════════════════════════════════ */
    .success-box {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       🦶 Footer
    ═══════════════════════════════════════════════════════════════════ */
    .footer {
        background: linear-gradient(135deg, #1A81A5 0%, #5AA7C0 100%);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-top: 40px;
        box-shadow: 0 -4px 20px rgba(26, 129, 165, 0.2);
    }
    
    .footer p {
        color: white;
        margin: 0;
        font-size: 14px;
        font-weight: 500;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* ═══════════════════════════════════════════════════════════════════
       📚 Retrieved Documents
    ═══════════════════════════════════════════════════════════════════ */
    .retrieved-doc {
        background: linear-gradient(135deg, #FFFFFF 0%, #F9FAFB 100%);
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    
    .retrieved-doc:hover {
        border-color: #5AA7C0;
        box-shadow: 0 4px 15px rgba(26, 129, 165, 0.1);
    }
    
    .doc-title {
        color: #1A81A5;
        font-weight: 700;
        font-size: 15px;
    }
    
    .doc-similarity {
        color: #10B981;
        font-size: 13px;
        font-weight: 600;
        float: right;
        background: #ECFDF5;
        padding: 3px 10px;
        border-radius: 15px;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       🔧 Diagnosis Checklist Styles
    ═══════════════════════════════════════════════════════════════════ */
    .diagnosis-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border-left: 5px solid #1A81A5;
        padding: 25px;
        margin: 20px 0;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(26, 129, 165, 0.12);
    }
    
    .diagnosis-title {
        color: #1A81A5;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 15px;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       🎯 Routing Result Box
    ═══════════════════════════════════════════════════════════════════ */
    .routing-box {
        background: linear-gradient(135deg, #FEF3C7 0%, #FDE68A 100%);
        border-left: 5px solid #F59E0B;
        padding: 25px;
        margin: 20px 0;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(245, 158, 11, 0.15);
    }
    
    .routing-title {
        color: #92400E;
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       👥 Team Badges
    ═══════════════════════════════════════════════════════════════════ */
    .team-badge {
        display: inline-block;
        padding: 12px 25px;
        border-radius: 30px;
        font-size: 16px;
        font-weight: 700;
        margin: 5px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.15);
    }
    
    .team-noc {
        background: linear-gradient(135deg, #DC2626, #EF4444);
        color: white;
    }
    
    .team-level2 {
        background: linear-gradient(135deg, #1A81A5, #5AA7C0);
        color: white;
    }
    
    .team-field {
        background: linear-gradient(135deg, #059669, #10B981);
        color: white;
    }
    
    /* Billing Team Badges */
    .team-payment {
        background: linear-gradient(135deg, #DC2626, #EF4444);
        color: white;
    }
    .team-billing {
        background: linear-gradient(135deg, #D97706, #F59E0B);
        color: white;
    }
    .team-account {
        background: linear-gradient(135deg, #7C3AED, #8B5CF6);
        color: white;
    }
    
    /* Service Request Team Badges */
    .team-upgrade {
        background: linear-gradient(135deg, #2563EB, #3B82F6);
        color: white;
    }
    .team-provision {
        background: linear-gradient(135deg, #059669, #10B981);
        color: white;
    }
    .team-quality {
        background: linear-gradient(135deg, #DC2626, #EF4444);
        color: white;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       ❓ Question Card
    ═══════════════════════════════════════════════════════════════════ */
    .question-card {
        background: white;
        border: 2px solid #E5E7EB;
        border-radius: 14px;
        padding: 18px;
        margin: 12px 0;
        transition: all 0.3s ease;
    }
    
    .question-card:hover {
        border-color: #5AA7C0;
        box-shadow: 0 4px 15px rgba(26, 129, 165, 0.1);
    }
    
    .question-card.answered {
        border-color: #10B981;
        background: linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%);
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       📻 Radio Buttons Custom Style
    ═══════════════════════════════════════════════════════════════════ */
    .stRadio > div {
        background: #F9FAFB;
        padding: 10px 15px;
        border-radius: 10px;
    }
    
    .stRadio > div > label {
        color: #374151;
        font-weight: 500;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       📊 Metrics
    ═══════════════════════════════════════════════════════════════════ */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 15px;
        border-radius: 12px;
        border-left: 4px solid #5AA7C0;
    }
    
    [data-testid="stMetricValue"] {
        color: #1A81A5;
        font-weight: 700;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       📝 Text Area
    ═══════════════════════════════════════════════════════════════════ */
    .stTextArea > div > div > textarea {
        border: 2px solid #E5E7EB;
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #5AA7C0;
        box-shadow: 0 0 0 3px rgba(90, 167, 192, 0.1);
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       ✨ Animations
    ═══════════════════════════════════════════════════════════════════ */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .entity-card, .question-card, .retrieved-doc {
        animation: fadeIn 0.3s ease-out;
    }
    
    /* ═══════════════════════════════════════════════════════════════════
       🎙️ Speaker Diarization Styles - Chat Interface
    ═══════════════════════════════════════════════════════════════════ */
    .diarization-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px 25px;
        border-radius: 16px 16px 0 0;
        margin-bottom: 0;
    }
    
    .diarization-header h3 {
        color: white;
        margin: 0;
        font-size: 18px;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .conversation-container {
        background: linear-gradient(180deg, #f0f2f5 0%, #e4e6eb 100%);
        border-radius: 0 0 16px 16px;
        padding: 25px;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-top: none;
    }
    
    .chat-message {
        display: flex;
        margin-bottom: 20px;
        animation: slideIn 0.3s ease-out;
    }
    
    .chat-message.agent {
        justify-content: flex-start;
    }
    
    .chat-message.customer {
        justify-content: flex-end;
    }
    
    .message-wrapper {
        max-width: 80%;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message.agent .message-wrapper {
        align-items: flex-start;
    }
    
    .chat-message.customer .message-wrapper {
        align-items: flex-end;
    }
    
    .speaker-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 6px;
    }
    
    .speaker-badge.agent {
        background: linear-gradient(135deg, #1A81A5, #5AA7C0);
        color: white;
    }
    
    .speaker-badge.customer {
        background: linear-gradient(135deg, #10B981, #34D399);
        color: white;
    }
    
    .message-bubble {
        padding: 14px 18px;
        border-radius: 18px;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        position: relative;
    }
    
    .chat-message.agent .message-bubble {
        background: white;
        color: #1f2937;
        border-bottom-left-radius: 4px;
        border: 1px solid #e5e7eb;
    }
    
    .chat-message.customer .message-bubble {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        border-bottom-right-radius: 4px;
        text-align: right;
        direction: rtl;
    }
    
    .message-text {
        word-wrap: break-word;
    }
    
    .message-time {
        font-size: 11px;
        margin-top: 6px;
        opacity: 0.7;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    
    .chat-message.agent .message-time {
        color: #6b7280;
    }
    
    .chat-message.customer .message-time {
        color: rgba(255,255,255,0.8);
        justify-content: flex-end;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .diarization-stats {
        display: flex;
        gap: 20px;
        margin-top: 15px;
        padding: 15px;
        background: white;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }
    
    .stat-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        color: #4b5563;
    }
    
    .stat-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }
    
    .stat-icon.agent {
        background: linear-gradient(135deg, #1A81A5, #5AA7C0);
        color: white;
    }
    
    .stat-icon.customer {
        background: linear-gradient(135deg, #10B981, #34D399);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Initialize LLM Components (cached)
# =============================================================================
@st.cache_resource
def init_groq_components():
    """Initialize Groq LLM components"""
    if GROQ_AVAILABLE:
        classifier = GroqClassifier(model="llama-3.3-70b-versatile")
        return classifier
    return None

# =============================================================================
# Header
# =============================================================================
# Logo on right (same as app_injaz.py)
col1, col2 = st.columns([2, 1])
with col2:
    st.image("assets/injaz_tech.png", width=180)

st.markdown("""
<div class="main-header">
    <h1>🎤 Speech to Text & Analysis</h1>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar Settings
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    use_groq_llm = st.checkbox(
        "🤖 Use Groq LLM (Recommended)",
        value=True,
        help="Use Llama 3.3 70B for better classification and routing"
    )
    
    st.markdown("---")
    st.markdown("### 📊 Pipeline Status")
    
    # Initialize Groq components
    try:
        groq_classifier = init_groq_components()
        
        if GROQ_AVAILABLE and use_groq_llm:
            st.success("✅ Groq LLM Ready")
            st.caption("🤖 Llama 3.3 70B")
        else:
            st.info("📝 Using Templates")
    except Exception as e:
        st.error(f"LLM Error: {e}")
        groq_classifier = None

# =============================================================================
# Main Content
# =============================================================================

# Config for Speech-to-Text
config = {
    "preprocess_audio": True,
    "correct_arabic": True,
    "use_gpt_correction": True,
    "extract_entities": True
}

# =============================================================================
# Step 1: Audio Upload
# =============================================================================
st.markdown('<p class="section-header">📤 Step 1: Upload Audio</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["mp3", "wav", "m4a", "flac", "ogg"],
    help="Supported: MP3, WAV, M4A, FLAC, OGG",
    label_visibility="collapsed"
)

audio_path = None
if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        audio_path = tmp.name
    
    st.audio(uploaded_file)
    st.success(f"✅ Uploaded: **{uploaded_file.name}**")

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# =============================================================================
# Step 2: Transcription & Processing
# =============================================================================
st.markdown('<p class="section-header">🔄 Step 2: Process & Analyze</p>', unsafe_allow_html=True)

if audio_path:
    if st.button("🚀 Process Audio", type="primary", use_container_width=True):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # ═══════════════════════════════════════════════════════════
            # Stage 1: Speech to Text with Segments
            # ═══════════════════════════════════════════════════════════
            status_text.text("🎤 Stage 1/4: Converting speech to text with timestamps...")
            progress_bar.progress(10)
            
            # Get transcription with segments for diarization
            try:
                whisper_result = transcribe_with_timestamps(audio_path, language="ar")
                transcription = whisper_result.get("text", "")
                segments = whisper_result.get("segments", [])
                
                # Apply Arabic text correction to the main transcription
                status_text.text("📝 Stage 1/4: Applying Arabic text correction...")
                try:
                    transcription = correct_arabic_text_gpt(transcription)
                except Exception as e:
                    print(f"⚠️ GPT correction failed: {e}")
                
                # ALWAYS apply dictionary corrections (catches errors GPT misses)
                transcription = apply_word_corrections(transcription)
                
                # Convert segments to the format expected by diarize_with_llm
                # Use dictionary correction for segments (faster than GPT for each)
                segment_list = []
                for seg in segments:
                    seg_text = seg.text if hasattr(seg, 'text') else seg.get('text', '')
                    # Apply fast dictionary-based correction
                    seg_text = apply_word_corrections(seg_text)
                    
                    segment_list.append({
                        "start": seg.start if hasattr(seg, 'start') else seg.get('start', 0),
                        "end": seg.end if hasattr(seg, 'end') else seg.get('end', 0),
                        "text": seg_text
                    })
                st.session_state.whisper_segments = segment_list
            except Exception as e:
                print(f"⚠️ Timestamp transcription failed, using basic: {e}")
                result = run(audio_path, config)
                if isinstance(result, dict):
                    transcription = result['text']
                else:
                    transcription = result
                segment_list = None
                st.session_state.whisper_segments = None
            
            # Extract entities from transcription
            entities = extract_call_center_entities(transcription)
            
            progress_bar.progress(20)
            st.session_state.transcription = transcription
            st.session_state.entities = entities
            
            # ═══════════════════════════════════════════════════════════
            # Stage 2: Speaker Diarization with LLM
            # ═══════════════════════════════════════════════════════════
            status_text.text("🎙️ Stage 2/4: Identifying speakers (Agent vs Customer)...")
            progress_bar.progress(30)
            
            # Use segments if available, otherwise use raw text
            if segment_list and len(segment_list) > 0:
                diarized_result = diarize_with_llm(segment_list)
            else:
                diarized_result = diarize_with_llm(transcription)
            
            st.session_state.diarized_conversation = diarized_result
            progress_bar.progress(40)
            
            # ═══════════════════════════════════════════════════════════
            # Stage 3: Classification (using Groq LLM if enabled)
            # ═══════════════════════════════════════════════════════════
            status_text.text("🏷️ Stage 3/4: Classifying complaint with AI...")
            progress_bar.progress(50)
            
            # Extract complaint from transcription
            # Use FULL transcription for classification (not truncated problem_statement)
            # This ensures classifier sees full context like "الانترنت فصل"
            problem_statement = entities.get('problem_statement', transcription)
            classification_text = transcription  # Use full text for accurate classification
            customer_name = entities.get('client_name', None)
            
            if groq_classifier:
                # Classify using Groq LLM
                category_code, category_name, confidence = groq_classifier.classify(classification_text)
                classification_method = "Groq LLM"
                
                st.session_state.classification = {
                    'code': category_code,
                    'name': category_name,
                    'confidence': confidence,
                    'method': classification_method
                }
                
                progress_bar.progress(60)
                
                # ═══════════════════════════════════════════════════════════
                # Stage 4: Team Routing
                # ═══════════════════════════════════════════════════════════
                status_text.text("🎯 Stage 4/4: Routing to appropriate team...")
                progress_bar.progress(85)
                
                # Generate unique escalation ID for this complaint
                import uuid
                escalation_id = f"ESC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
                st.session_state.escalation_id = escalation_id
                
                # ═══════════════════════════════════════════════════════════
                # Team Routing (Billing & Service Request)
                # ═══════════════════════════════════════════════════════════
                team_routing = None
                if category_code == 'billing':
                    billing_router = get_billing_router()
                    team_routing = billing_router.classify(problem_statement)
                elif category_code == 'service_request':
                    sr_router = get_service_request_router()
                    team_routing = sr_router.classify(problem_statement)
                elif category_code == 'internet':
                    # LLM-based internet diagnosis (automatic)
                    try:
                        diagnosis_flow = InternetDiagnosisFlow()
                        llm_result = diagnosis_flow.run_flow(transcription)
                        st.session_state.diagnosis_routing = llm_result.get('التحليل', {})
                        st.session_state.diagnosis_json = llm_result
                    except Exception as diag_e:
                        st.session_state.diagnosis_routing = {}
                        print(f"Internet diagnosis error: {diag_e}")
                
                st.session_state.team_routing = team_routing
            
            progress_bar.progress(100)
            status_text.text("✅ Processing complete!")
            
            # Success message
            st.markdown("""
            <div class="success-box">
                ✅ Audio processed successfully!
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            import traceback
            st.code(traceback.format_exc())
        
        finally:
            # Cleanup temp file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass

# =============================================================================
# Display Results
# =============================================================================
if 'transcription' in st.session_state and st.session_state.transcription:
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # Speaker Diarization (Agent vs Customer) - Clean Chat Style
    # ═══════════════════════════════════════════════════════════════
    if 'diarized_conversation' in st.session_state and st.session_state.diarized_conversation:
        # Merge consecutive messages from the same speaker
        diarized = merge_consecutive_speaker_messages(st.session_state.diarized_conversation)
        
        # Header - Same blue gradient as main header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1A81A5 0%, #5AA7C0 100%); padding: 18px 25px; border-radius: 16px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(26, 129, 165, 0.3);">
            <h3 style="color: white; margin: 0; font-size: 18px; font-weight: 600;">🎙️ Speaker Diarization</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Chat container with light gradient background
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e8f4f8 50%, #fce7f3 100%); padding: 30px 20px; border-radius: 16px; margin-bottom: 20px;">
        <p style="text-align: center; color: #6B7280; font-size: 18px; margin: 0 0 20px 0; font-weight: 700;">💬 Conversation between Customer and Agent</p>
        """, unsafe_allow_html=True)
        
        # Display each message
        for turn in diarized:
            speaker = turn.get('speaker', 'Unknown')
            text = turn.get('text', '')
            
            if speaker == 'Customer':
                # Customer message - LEFT side, white bubble with avatar
                st.markdown(f"""
                <div style="display: flex; align-items: flex-start; margin-bottom: 25px; gap: 12px;">
                    <div style="width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #667eea, #764ba2); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                        <span style="color: white; font-size: 18px;">👤</span>
                    </div>
                    <div style="max-width: 75%;">
                        <div style="background: white; padding: 14px 18px; border-radius: 20px; border-top-left-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); direction: rtl; text-align: right;">
                            <div style="color: #1f2937; font-size: 15px; line-height: 1.6;">{text}</div>
                        </div>
                        <div style="color: #9ca3af; font-size: 11px; margin-top: 6px; margin-left: 5px;">العميل</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Agent message - RIGHT side, blue bubble with icon beside text
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; align-items: flex-start; margin-bottom: 25px; gap: 12px;">
                    <div style="max-width: 75%;">
                        <div style="background: linear-gradient(135deg, #2563EB, #3B82F6); color: white; padding: 14px 18px; border-radius: 20px; border-top-right-radius: 4px; box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25); direction: rtl;">
                            <div style="font-size: 15px; line-height: 1.6;">{text}</div>
                        </div>
                        <div style="color: #6B7280; font-size: 11px; margin-top: 6px; text-align: right;">موظف خدمة العملاء</div>
                    </div>
                    <div style="width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(135deg, #2563EB, #3B82F6); display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                        <span style="color: white; font-size: 18px;">👨‍💼</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Close container
        st.markdown("</div>", unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # Classification Result
    # ═══════════════════════════════════════════════════════════════
    if 'classification' in st.session_state:
        st.markdown('<p class="section-header">🏷️ Complaint Classification</p>', unsafe_allow_html=True)
        
        classification = st.session_state.classification
        
        badge_class = {
            'internet': 'badge-internet',
            'billing': 'badge-billing',
            'service_request': 'badge-service'
        }.get(classification['code'], 'badge-internet')
        
        st.markdown(f"""
        <div class="entity-card">
            <div class="entity-label">📂 Category</div>
            <div class="entity-value">
                <span class="classification-badge {badge_class}">{classification['name']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════
        # Team Routing Display (Billing & Service Request)
        # ═══════════════════════════════════════════════════════════════
        if classification['code'] in ('billing', 'service_request') and 'team_routing' in st.session_state and st.session_state.team_routing:
            routing = st.session_state.team_routing
            
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            
            # Team routing header
            if classification['code'] == 'billing':
                header_text = "توجيه شكوى الفواتير"
                header_icon = "💰"
            else:
                header_text = "توجيه طلب الخدمة"
                header_icon = "🛠️"
            
            st.markdown(f"""
            <div style="margin: 25px 0 20px 0; text-align: right; direction: rtl;">
                <h3 style="color: #1A81A5; margin: 0; font-size: 20px; font-weight: 600;">
                    {header_icon} {header_text}
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Team color mapping
            team_colors = {
                "PaymentSupport": ("#DC2626", "💳"),
                "BillingSupport": ("#D97706", "🧾"),
                "AccountManagement": ("#7C3AED", "👤"),
                "PlanUpgradeManagement": ("#2563EB", "📦"),
                "ServiceProvisioning": ("#059669", "🔌"),
                "ServiceQualitySupport": ("#DC2626", "📡"),
            }
            color, icon = team_colors.get(routing.get('team', ''), ("#1A81A5", "⚪"))
            
            st.markdown(f"""
            <div style="background: #f8f9fa; border-radius: 10px; padding: 20px; margin: 15px 0; border-right: 4px solid {color};">
                <p style="color: #666; font-size: 13px; margin: 0 0 8px 0; text-align: right; direction: rtl;">الفريق المسؤول</p>
                <p style="color: {color}; font-size: 18px; font-weight: 600; margin: 0; text-align: right; direction: rtl;">{icon} {routing.get('team', '')} - {routing.get('team_ar', '')}</p>
                <p style="color: #555; font-size: 14px; margin: 12px 0 0 0; text-align: right; direction: rtl;">📋 {routing.get('action', '')}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════════════════
        # Internet Diagnosis (Only for Internet Problems) - LLM Only
        # ═══════════════════════════════════════════════════════════════
        if classification['code'] == 'internet' and 'diagnosis_routing' in st.session_state and st.session_state.diagnosis_routing:
            st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
            
            st.markdown("""
            <div style="margin: 25px 0 20px 0; text-align: right; direction: rtl;">
                <h3 style="color: #1A81A5; margin: 0; font-size: 20px; font-weight: 600;">
                    🔧 تشخيص مشكلة الإنترنت
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            routing = st.session_state.diagnosis_routing
            
            # Get team display info
            team_key = routing.get('team', routing.get('الفريق_المناسب', ''))
            team_ar = routing.get('team_ar', routing.get('اسم_الفريق_بالعربي', ''))
            action = routing.get('action', routing.get('الإجراء_المقترح', ''))
            
            team_info = {
                "NOC": ("🔴", "فريق الشبكات", "#DC2626"),
                "Level 2": ("🔵", "الدعم الفني المتقدم", "#1A81A5"),
                "Field Support": ("🟢", "الدعم الميداني", "#059669")
            }
            emoji, team_name_display, team_color = team_info.get(team_key, ("⚪", team_ar or "غير محدد", "#1A81A5"))
            
            st.markdown(f"""
            <div style="background: #f8f9fa; border-radius: 10px; padding: 20px; margin: 15px 0; border-right: 4px solid {team_color}; text-align: right; direction: rtl;">
                <p style="color: #666; font-size: 13px; margin: 0 0 8px 0;">نتيجة التوجيه</p>
                <p style="color: {team_color}; font-size: 18px; font-weight: 600; margin: 0;">{emoji} {team_key} - {team_name_display}</p>
                <p style="color: #555; font-size: 14px; margin: 12px 0 0 0;">{action}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # Extracted Information (After Team Determination)
    # ═══════════════════════════════════════════════════════════════
    if 'entities' in st.session_state:
        st.markdown('<p class="section-header">📋 Extracted Information</p>', unsafe_allow_html=True)
        
        entities = st.session_state.entities
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="entity-card">
                <div class="entity-label">👤 Client Name</div>
                <div class="entity-value">{entities.get('client_name', 'Not found') or 'Not found'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="entity-card">
                <div class="entity-label">📞 Phone Number</div>
                <div class="entity-value">{entities.get('phone_number', 'Not found') or 'Not found'}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="entity-card">
                <div class="entity-label">👨‍💼 Agent Name</div>
                <div class="entity-value">{entities.get('agent_name', 'Not found') or 'Not found'}</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)
    
    # ═══════════════════════════════════════════════════════════════
    # Export Options - Excel Only
    # ═══════════════════════════════════════════════════════════════
    if st.button("📊 Save to Excel", type="primary", use_container_width=True):
            import pandas as pd
            import uuid as _uuid
            
            # Get category code
            category_code = st.session_state.get('classification', {}).get('code', '')
            
            # Get diagnosis data if available (internet only)
            diagnosis_data = st.session_state.get('diagnosis_json', {})
            routing_data = st.session_state.get('diagnosis_routing', {})
            
            # Generate a ticket ID for ALL categories
            ticket_id = diagnosis_data.get('رقم_التذكرة', '')
            if not ticket_id:
                ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d')}-{_uuid.uuid4().hex[:6].upper()}"
            
            # Base data for all categories
            data = {
                'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
                'Ticket ID': [ticket_id],
                'Client Name': [st.session_state.entities.get('client_name', '')],
                'Phone Number': [st.session_state.entities.get('phone_number', '')],
                'Category': [st.session_state.get('classification', {}).get('name', '')],
                'Transcription': [st.session_state.transcription],
                'Escalation ID': [st.session_state.get('escalation_id', '')],
            }
            
            # Add Agent Name only if it exists
            agent_name = st.session_state.entities.get('agent_name', '')
            if agent_name:
                data['Agent Name'] = [agent_name]
            
            # Add internet-specific diagnosis fields ONLY for internet category
            if category_code == 'internet':
                # LLM-based diagnosis - extract from JSON output
                analysis = diagnosis_data.get('التحليل', {})
                data['Assigned Team'] = [analysis.get('الفريق_المناسب', routing_data.get('team', ''))]
                data['Team (AR)'] = [analysis.get('اسم_الفريق_بالعربي', routing_data.get('team_ar', ''))]
                data['Action Required'] = [diagnosis_data.get('الإجراء_المقترح', routing_data.get('action', ''))]
            
            # Add team routing fields for billing and service_request
            if category_code in ('billing', 'service_request'):
                team_routing = st.session_state.get('team_routing', {})
                if team_routing:
                    data['Assigned Team'] = [team_routing.get('team', '')]
                    data['Team (AR)'] = [team_routing.get('team_ar', '')]
                    data['Action Required'] = [team_routing.get('action', '')]
            
            df = pd.DataFrame(data)
            excel_file = os.path.join(tempfile.gettempdir(), f'rag_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
            df.to_excel(excel_file, index=False)
            
            st.success(f"✅ Saved successfully!")
            
            with open(excel_file, 'rb') as f:
                st.download_button(
                    label="📥 Download Excel",
                    data=f,
                    file_name=os.path.basename(excel_file),
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

# =============================================================================
# Footer
# =============================================================================
st.markdown("""
<div class="footer">
    <p>© 2026 Injaz Tech</p>
</div>
""", unsafe_allow_html=True)
