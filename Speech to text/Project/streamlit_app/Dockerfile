FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY streamlit_app/requirements.txt .
RUN pip install -r requirements.txt

COPY app_rag.py .
COPY LLM_turns.py .
COPY Complaints/ ./Complaints/
COPY modules/ ./modules/
COPY shared/ ./shared/
COPY assets/ ./assets/

EXPOSE 8501

CMD ["streamlit", "run", "app_rag.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
