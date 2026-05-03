"""
Local API - Speech-to-Text using Local Whisper & Sentence Transformers
No OpenAI API key required - runs completely offline
Port: 8001
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import os
import tempfile
import logging
from datetime import datetime

# Import local modules (all from test2 folder)
from local_whisper import transcribe_local
from local_analyzer import LocalAnalyzer
from audio_preprocessor_local import preprocess_audio_simple
from arabic_corrector_local import correct_arabic_text
from extract_local import extract_entities_local, extract_entities_with_gpt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# FastAPI App Configuration
# =============================================================================
app = FastAPI(
    title="Local Speech-to-Text API",
    description="Arabic Call Center Transcription - Local Whisper & Sentence Transformers (No API key required)",
    version="2.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Initialize analyzer (loads model at startup)
analyzer = None

@app.on_event("startup")
async def startup_event():
    """Load models at startup"""
    global analyzer
    logger.info("🚀 Starting Local Speech-to-Text API...")
    
    # Load Sentence Transformers model
    logger.info("🔄 Loading Sentence Transformers model...")
    analyzer = LocalAnalyzer()
    
    # Pre-load Whisper model at startup (not at first request)
    logger.info("🔄 Loading Whisper Medium model (this takes 1-2 minutes on CPU)...")
    from local_whisper import get_model
    get_model("small")  # Pre-load small model
    
    logger.info("✅ All models loaded! API ready.")


# =============================================================================
# Response Models
# =============================================================================
class TranscriptionResponse(BaseModel):
    transcription: str
    entities: Dict[str, Any]
    status: str = "new"
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    whisper_model: str
    analyzer: str
    timestamp: str


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Local Speech-to-Text API",
        "status": "running",
        "description": "Uses Local Whisper Large + Sentence Transformers (No API key required)",
        "endpoints": {
            "health": "/health",
            "transcribe": "/transcribe (POST)",
            "docs": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        whisper_model="large (local)",
        analyzer="Sentence Transformers (paraphrase-multilingual-MiniLM-L12-v2)",
        timestamp=datetime.now().isoformat()
    )

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    preprocess_audio: bool = True,
    correct_arabic: bool = True,
    extract_entities: bool = True,
    analyze_category: bool = True,
    whisper_model: str = "small"
):
    """
    Transcribe audio file using LOCAL Whisper and extract entities
    
    Parameters:
    - file: Audio file (mp3, wav, m4a, flac, ogg)
    - preprocess_audio: Apply audio preprocessing (default: True)
    - correct_arabic: Apply Arabic text correction (default: True)
    - extract_entities: Extract entities from transcription (default: True)
    - analyze_category: Analyze category using Sentence Transformers (default: True)
    - whisper_model: Whisper model size - tiny, base, small, medium (default: medium)
    
    Returns:
    - Transcription text
    - Extracted entities (client name, phone number)
    - Category classification
    - Sentiment analysis
    """
    
    # Validate file type
    allowed_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format '{file_ext}'. Allowed formats: {', '.join(allowed_extensions)}"
        )
    
    temp_file_path = None
    preprocessed_path = None
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            content = await file.read()
            tmp.write(content)
            temp_file_path = tmp.name
        
        logger.info(f"📁 Processing file: {file.filename} ({len(content)} bytes)")
        
        audio_path = temp_file_path
        
        # Apply audio preprocessing if enabled
        if preprocess_audio:
            logger.info("🔧 Applying audio preprocessing...")
            preprocessed_path = preprocess_audio_simple(temp_file_path)
            if preprocessed_path:
                audio_path = preprocessed_path
        
        # Transcribe using LOCAL Whisper
        logger.info(f"🎤 Transcribing with local Whisper {whisper_model}...")
        transcription = transcribe_local(audio_path, language="ar", model_name=whisper_model)
        logger.info(f"🗒️ Transcription (raw): {transcription[:200]}")
        
        # Apply Arabic text correction
        if correct_arabic:
            logger.info("📝 Applying Arabic text correction...")
            transcription = correct_arabic_text(transcription)
            logger.info(f"🛠️ Transcription (corrected): {transcription[:200]}")
        
        # Extract entities (regex-based) and optionally with OpenAI fallback
        entities = {}
        if extract_entities:
            logger.info("🔍 Extracting entities (regex)...")
            entities = extract_entities_local(transcription)
            # If name missing and OpenAI key present, try GPT extractor
            if (not entities.get("client_name")) and os.getenv("OPENAI_API_KEY"):
                logger.info("🤖 Falling back to OpenAI extractor for entities...")
                try:
                    gpt_entities = extract_entities_with_gpt(transcription)
                    # merge results if available
                    if gpt_entities.get("client_name"):
                        entities["client_name"] = gpt_entities.get("client_name")
                    if gpt_entities.get("phone_number"):
                        entities["phone_number"] = gpt_entities.get("phone_number")
                except Exception as e:
                    logger.warning(f"GPT extractor failed: {e}")
        
        # Analyze category and sentiment
        if analyze_category and analyzer:
            logger.info("🤖 Analyzing category and sentiment...")
            analysis = analyzer.analyze_text(transcription)
            entities['category'] = analysis['category']['label']
            entities['sentiment'] = analysis['sentiment']['label']
        
        # Prepare response
        response_data = TranscriptionResponse(
            transcription=transcription,
            entities=entities,
            status="completed",
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Successfully processed: {file.filename}")
        
        return response_data
    
    except Exception as e:
        logger.error(f"❌ Error processing {file.filename}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio file: {str(e)}"
        )
    
    finally:
        # Clean up temporary files
        for path in [temp_file_path, preprocessed_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file: {cleanup_error}")


# =============================================================================
# Run Server
# =============================================================================
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("🚀 Local Speech-to-Text API")
    print("=" * 60)
    print("✅ No OpenAI API key required!")
    print("📦 Using: Whisper Large (local) + Sentence Transformers")
    print("🌐 Port: 8001")
    print("=" * 60)
    
    uvicorn.run(
        "local_api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    )
