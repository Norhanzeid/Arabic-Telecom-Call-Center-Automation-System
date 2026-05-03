FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install -r requirements.txt

COPY Complaints/ ./Complaints/
COPY modules/ ./modules/
COPY shared/ ./shared/
COPY LLM_turns.py .

EXPOSE 8000

CMD ["uvicorn", "Complaints.api:app", "--host", "0.0.0.0", "--port", "8000"]
