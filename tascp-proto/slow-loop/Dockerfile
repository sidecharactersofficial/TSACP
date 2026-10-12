FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# We need sqlite3 in the system to interact with the database if we want to debug
RUN apt-get update && apt-get install -y sqlite3 && rm -rf /var/lib/apt/lists/*
CMD ["python", "worker.py"]
