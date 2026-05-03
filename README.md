# Arabic Telecom Call Center Automation System

An advanced AI-powered call center automation system that converts Arabic speech to text, classifies complaints, extracts entities, and intelligently routes requests using LLM-based processing.

> **📂 Project Location**: [Speech to text/Project/](Speech%20to%20text/Project)

## 🎯 Quick Start

This repository contains the complete implementation of an intelligent Arabic call center automation system.

### Key Features:
- 🎙️ **Speech-to-Text**: Arabic audio transcription with OpenAI Whisper
- 🤖 **LLM Classification**: Intelligent complaint categorization using Groq Llama 3.3
- 📋 **Entity Extraction**: Automatic customer information extraction
- 🔀 **Smart Routing**: Route complaints to Billing, Internet Support, or Service Requests
- 🗣️ **Speaker Diarization**: Identify who's speaking in calls
- 🌐 **Web Interface**: Streamlit-based dashboard + FastAPI REST API

---

## 📖 Full Documentation

For complete setup instructions, API documentation, architecture details, and more, see:

[**📘 Full README → Speech to text/Project/README.md**](Speech%20to%20text/Project/README.md)

---

## 🚀 Quick Installation

### Prerequisites
- Python 3.8+
- FFmpeg
- API Keys: Groq + OpenAI

### Setup
```bash
cd "Speech to text/Project"
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run
```bash
# Web Dashboard (Streamlit)
streamlit run app_rag.py

# REST API
python -m uvicorn Complaints.api:app --reload

# Docker (both services)
docker-compose up --build
```

---

## 📁 Project Structure

```
Speech to text/Project/
├── README.md                          # Full documentation
├── app_rag.py                         # Streamlit web interface
├── Complaints/
│   ├── api.py                         # FastAPI server
│   ├── classifier.py                  # LLM classifier
│   ├── billing_router.py              # Billing routing
│   ├── service_request_router.py      # Service routing
│   └── internet_diagnosis_flow.py     # Internet troubleshooting
├── modules/speech_to_text/            # STT & processing
├── prompts/                           # LLM system prompts
├── test_cases/                        # Tests
└── requirements.txt
```

---

## 🔗 Links

- **Full Documentation**: [Speech to text/Project/README.md](Speech%20to%20text/Project/README.md)
- **API Server**: Run with `python -m uvicorn Complaints.api:app` (http://localhost:8000)
- **Web Dashboard**: Run with `streamlit run app_rag.py` (http://localhost:8501)

---

## 📝 Environment Setup

Create a `.env` file in `Speech to text/Project/`:
```env
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key
```

Get keys from:
- **Groq**: https://console.groq.com
- **OpenAI**: https://platform.openai.com
