"""
Arabic Call Center API
"""

import os
import sys

# Load environment variables FIRST before any other imports
from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(env_path)

# Add Project directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional
import tempfile
import shutil
from datetime import datetime

from groq import Groq
from .billing_router import get_billing_router
from .service_request_router import get_service_request_router
from .internet_diagnosis_flow import InternetDiagnosisFlow
from prompt_loader import load_prompt

# Import modules from parent directory
try:
    from modules.speech_to_text.extract import extract_call_center_entities
    from modules.speech_to_text.openai_whisper import transcribe_with_timestamps, validate_audio_format
    from LLM_turns import diarize_with_llm
    STT_AVAILABLE = True
except ImportError as e:
    print(f"STT modules not available: {e}")
    STT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════
# Groq Classifier (same as app_rag.py)
# ═══════════════════════════════════════════════════════════════════

class GroqClassifier:
    """LLM-based Arabic complaint classifier using Groq"""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.model = model
        self.system_prompt = load_prompt("complaints_api_classifier_system")

    def classify(self, complaint_text: str) -> tuple:
        """Classify complaint using Groq LLM"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"صنف هذه الشكوى:\n\n{complaint_text}"}
                ],
                temperature=0,  # deterministic
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


# ═══════════════════════════════════════════════════════════════════
# FastAPI App
# ═══════════════════════════════════════════════════════════════════

app = FastAPI(title="Arabic Call Center")

# Pipeline components (lazy initialization)
_groq_classifier: Optional[GroqClassifier] = None
_diagnosis_flow: Optional[InternetDiagnosisFlow] = None
_complaints_processed: int = 0


def get_groq_classifier() -> GroqClassifier:
    """Get Groq classifier instance"""
    global _groq_classifier
    if _groq_classifier is None:
        _groq_classifier = GroqClassifier()
    return _groq_classifier


def get_diagnosis_flow() -> InternetDiagnosisFlow:
    """Get diagnosis flow instance"""
    global _diagnosis_flow
    if _diagnosis_flow is None:
        _diagnosis_flow = InternetDiagnosisFlow()
    return _diagnosis_flow


# ═══════════════════════════════════════════════════════════════════
# Single Endpoint - Complete Pipeline
# ═══════════════════════════════════════════════════════════════════

@app.post("/process")
async def process_full_pipeline(
    file: UploadFile = File(...),
    extract_entities: bool = True,
    diarize: bool = True,
    language: str = None  # None for auto-detect, 'ar' for Arabic
):
    """
    Complete Pipeline - Audio to Classification and Routing
    
    Steps:
    1. Speech-to-Text (transcribe audio with segments)
    2. Entity Extraction (client name, phone, agent name)
    3. Speaker Diarization (agent vs customer using segments)
    4. Complaint Classification
    5. Team Routing
    6. Internet Diagnosis (for internet issues)
    
    Input: Audio file (mp3, wav, m4a, mp4, ogg, webm, flac)
    """
    global _complaints_processed
    
    try:
        if not STT_AVAILABLE:
            raise HTTPException(status_code=503, detail="Speech-to-Text not available")
        
        # Validate audio format
        if not validate_audio_format(file.filename):
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported audio format. Supported: mp3, wav, m4a, mp4, ogg, webm, flac"
            )
        
        # Step 1: Speech-to-Text with segments
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        try:
            # Get full response with segments for better diarization
            whisper_response = transcribe_with_timestamps(temp_path, language=language)
            transcription = whisper_response["text"]
            segments = whisper_response.get("segments", [])
            detected_language = whisper_response.get("language", "unknown")
            duration = whisper_response.get("duration", 0.0)
            
            if not transcription or not transcription.strip():
                raise HTTPException(status_code=400, detail="Failed to transcribe audio - no speech detected")
                
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        # Step 2: Entity Extraction 
        entities = None
        if extract_entities:
            try:
                entities = extract_call_center_entities(transcription)
            except Exception as e:
                print(f"Entity extraction error: {e}")
                entities = {"error": f"Entity extraction failed: {str(e)}"}
        
        # Step 3: Speaker Diarization using segments for better accuracy
        diarized_conversation = None
        if diarize:
            try:
                # Use segments if available, otherwise fall back to text
                diarize_input = segments if segments else transcription
                diarized_conversation = diarize_with_llm(diarize_input)
                if not diarized_conversation:
                    print("Diarization returned None, falling back to text-only")
                    diarized_conversation = diarize_with_llm(transcription)
            except Exception as e:
                print(f"Diarization error: {e}")
                diarized_conversation = {"error": f"Diarization failed: {str(e)}"}
        
        # Step 4: Classification
        classifier = get_groq_classifier()
        category_code, category_name, confidence = classifier.classify(transcription)
        
        # Step 5: Team Routing
        team_routing = None
        diagnosis_result = None
        
        try:
            if category_code == "internet":
                flow = get_diagnosis_flow()
                diagnosis_result = flow.run_flow(transcription)
                team_routing = diagnosis_result.get("التحليل", {})
            elif category_code == "billing":
                router = get_billing_router()
                team_routing = router.classify(transcription)
            elif category_code == "service_request":
                router = get_service_request_router()
                team_routing = router.classify(transcription)
        except Exception as e:
            print(f"Team routing error: {e}")
            team_routing = {"error": f"Team routing failed: {str(e)}"}
        
        _complaints_processed += 1
        
        # Build comprehensive response
        response = {
            "processing_id": f"PRC-{datetime.now().strftime('%Y%m%d')}-{_complaints_processed:04d}",
            "timestamp": datetime.now().isoformat(),
            "input": {
                "type": "audio",
                "filename": file.filename,
                "detected_language": detected_language,
                "duration_seconds": duration,
                "audio_format": suffix,
                "text_length": len(transcription)
            },
            "transcription": transcription,
            "classification": {
                "category_code": category_code,
                "category_name": category_name,
                "confidence": confidence
            }
        }
        
        # Add entities if extracted
        if entities:
            response["entities"] = entities
        
        # Add diarization if performed
        if diarized_conversation:
            if isinstance(diarized_conversation, list):
                response["diarization"] = {
                    "conversation": diarized_conversation,
                    "turns_count": len(diarized_conversation),
                    "method": "segments" if segments else "text_only"
                }
            else:
                response["diarization"] = diarized_conversation
        
        # Add team routing
        if team_routing:
            if isinstance(team_routing, dict) and "error" not in team_routing:
                response["team_routing"] = {
                    "team": team_routing.get("team", team_routing.get("الفريق_المناسب")),
                    "team_ar": team_routing.get("team_ar", team_routing.get("اسم_الفريق_بالعربي")),
                    "action": team_routing.get("action", team_routing.get("الإجراء_المقترح"))
                }
            else:
                response["team_routing"] = team_routing
        
        # Add full diagnosis for internet issues
        if diagnosis_result:
            response["internet_diagnosis"] = diagnosis_result
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


