FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY models.py .
COPY server/app.py .
COPY server/environment.py .
COPY server/engine.py .
COPY server/graders.py .
COPY server/task_data.py .
COPY server/static/ ./static/

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
