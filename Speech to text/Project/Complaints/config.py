"""
Configuration for Complaint RAG System
إعدادات نظام RAG لشكاوى العملاء
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
VECTOR_DB_DIR = BASE_DIR / "vector_databases"

# Vector Database paths - قاعدة بيانات منفصلة لكل فئة
INTERNET_DB_PATH = VECTOR_DB_DIR / "internet_vector_db"
BILLING_DB_PATH = VECTOR_DB_DIR / "billing_vector_db"
SERVICE_REQUEST_DB_PATH = VECTOR_DB_DIR / "service_request_vector_db"

# Categories - الفئات المدعومة
CATEGORIES = {
    "internet": "مشكلة نت",
    "billing": "مشكلة فواتير",
    "service_request": "مشكلة طلب خدمة"
}

# Embedding model
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Groq API Key (set via .env file)
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# LLM settings (Groq - Llama 3.3 70B)
LLM_MODEL = "llama-3.3-70b-versatile"  # Groq model
MAX_TOKENS = 800
TEMPERATURE = 0  # 0 for deterministic responses

# Retrieval settings
TOP_K_RESULTS = 3  # عدد النتائج المسترجعة
SIMILARITY_THRESHOLD = 0.5

# Create directories
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
INTERNET_DB_PATH.mkdir(parents=True, exist_ok=True)
BILLING_DB_PATH.mkdir(parents=True, exist_ok=True)
SERVICE_REQUEST_DB_PATH.mkdir(parents=True, exist_ok=True)
