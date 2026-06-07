# Arabic Telecom Call Center Automation System

An advanced AI-powered call center automation system that converts Arabic speech to text, classifies complaints, extracts entities, and intelligently routes requests using LLM-based processing.

## 🎯 Overview

This project integrates cutting-edge AI technologies to automate Arabic call center workflows:
- **Speech-to-Text**: Transcribe Arabic audio calls using OpenAI Whisper
- **Call Analysis**: Diarize speakers and extract key information using LLM
- **Smart Routing**: Intelligently classify and route complaints to appropriate departments (billing, service requests, internet issues)
- **Entity Extraction**: Automatically extract customer names, phone numbers, and complaint details
- **LLM-Based Processing**: Leverage Groq's Llama 3.3 model for intelligent classification and analysis

---

## ✨ Key Features

### 1. **Speech-to-Text Processing**
- Transcribe Arabic audio with automatic timestamps
- Real-time audio preprocessing and noise reduction
- Support for multiple audio formats (MP3, WAV, OGG, M4A)

### 2. **Intelligent Complaint Classification**
- Multi-category classification (Billing, Internet, Service Requests)
- Arabic text processing and normalization
- LLM-powered classification using Groq

### 3. **Entity Extraction**
- Automatic extraction of customer details:
  - Names
  - Phone numbers
  - Account information
  - Issue descriptions
- Support for both GPT and local processing

### 4. **Smart Call Routing**
- **Billing Router**: Route billing-related complaints
- **Service Request Router**: Handle service requests
- **Internet Diagnosis Flow**: Diagnose and troubleshoot internet issues with guided workflows

### 5. **Call Center Features**
- Speaker diarization (identify who's speaking)
- Turn-by-turn analysis
- Multi-turn conversation tracking
- Arabic text correction and normalization

### 6. **Web Interface**
- Streamlit-based web dashboard
- Real-time processing
- Visual call analysis

---

## 🏗️ Architecture

```
Speech-to-Text → Transcription & Diarization
                 ↓
          Entity Extraction
                 ↓
          LLM Classification
                 ↓
    ┌────────────┼────────────┐
    ↓            ↓            ↓
Billing      Internet     Service
Router     Diagnosis      Router
    ↓            ↓            ↓
  [Routing & Response Generation]
```

---

## 🎥 Demo Video

Watch the project in action:

[**▶️ Watch Demo Video on Google Drive**](https://drive.google.com/file/d/1ik5h897vGwRcm1NGKDYlIOs0biRWkZok/view?usp=sharing)

---

## 📋 Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)
- API Keys:
  - **Groq API Key** (for LLM classification)
  - **OpenAI API Key** (for entity extraction)
- Docker & Docker Compose (optional)

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Norhanzeid/Arabic-Telecom-Call-Center-Automation-System.git
cd Speech\ to\ text/Project
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install FFmpeg
**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
Download from [FFmpeg Official Site](https://ffmpeg.org/download.html)

---

## ⚙️ Configuration

### 1. Create `.env` File
```bash
cp .env.example .env  # if available, or create manually
```

### 2. Add API Keys to `.env`
```env
# Groq API Key (for LLM)
GROQ_API_KEY=your_groq_api_key_here

# OpenAI API Key (for entity extraction)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Model configurations
GROQ_MODEL=llama-3.3-70b-versatile
OPENAI_MODEL=gpt-4
```

### 3. Get API Keys
- **Groq**: https://console.groq.com
- **OpenAI**: https://platform.openai.com/api-keys

---

## 🎮 Running the Application

### Option 1: Streamlit Web Interface
```bash
streamlit run app_rag.py
```
Access at: `http://localhost:8501`

### Option 2: FastAPI REST API
```bash
python -m uvicorn Complaints.api:app --host 0.0.0.0 --port 8000 --reload
```
Access at: `http://localhost:8000`
API Docs: `http://localhost:8000/docs`

### Option 3: Docker Compose (Both Services)
```bash
docker-compose up --build
```
- Streamlit: http://localhost:8501
- API: http://localhost:8000

---


---

## 📁 Project Structure

```
Project/
├── app_rag.py                          # Streamlit web interface
├── llm_analyzer.py                     # LLM analysis utilities
├── LLM_turns.py                        # Speaker diarization
├── prompt_loader.py                    # Load system prompts
├── template_injaz.py                   # Email template generator
│
├── modules/
│   └── speech_to_text/
│       ├── task.py                     # Main STT pipeline
│       ├── extract.py                  # Entity extraction
│       ├── openai_whisper.py           # Whisper transcription
│       ├── arabic_corrector.py         # Arabic text correction
│       └── audio_preprocessor.py       # Audio preprocessing
│
├── Complaints/
│   ├── api.py                          # FastAPI application
│   ├── classifier.py                   # Complaint classifier
│   ├── billing_router.py               # Billing department routing
│   ├── service_request_router.py       # Service request routing
│   ├── internet_diagnosis_flow.py      # Internet troubleshooting flow
│   └── config.py                       # Configuration
│
├── prompts/                            # LLM system prompts
│   ├── classifier_system.txt
│   ├── billing_router_system.txt
│   ├── service_request_router_system.txt
│   ├── internet_diagnosis_*.txt
│   └── ...
│
├── test_cases/                         # Unit and integration tests
│   ├── test_extraction.py
│   ├── test_billing_router.py
│   ├── test_api_full_pipeline.py
│   └── ...
│
├── test2/                              # Local processing alternatives
│   ├── local_whisper.py
│   ├── extract_local.py
│   └── ...
│
├── streamlit_app/
│   ├── Dockerfile                      # Streamlit container
│   └── requirements.txt
│
├── api/
│   ├── Dockerfile                      # API container
│   └── requirements.txt
│
├── shared/
│   ├── config.py                       # Shared configuration
│   └── __init__.py
│
├── docker-compose.yml                  # Multi-container setup
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```










