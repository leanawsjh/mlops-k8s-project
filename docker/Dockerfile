FROM python:3.10-slim

WORKDIR /app

COPY app/ ./app/
COPY models/saved/ ./models/saved/
COPY app/requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app/app

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
