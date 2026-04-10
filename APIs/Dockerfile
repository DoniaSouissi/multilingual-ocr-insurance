FROM python:3.10-slim
 
WORKDIR /app
 
# Install OS-level dependencies

RUN apt-get update && \

    apt-get install -y libgl1 libglib2.0-0 curl gnupg && \

    rm -rf /var/lib/apt/lists/*
 
# Install Ollama

RUN curl -fsSL https://ollama.com/install.sh | bash
 
# Copy project files into the container

COPY . .
 
# Install Python dependencies

RUN pip install --upgrade pip && pip install -r requirements.txt
 
# Expose FastAPI port

EXPOSE 8000
 
# Start Ollama in background and then run FastAPI

CMD bash -c "ollama serve & uvicorn main:app --host 0.0.0.0 --port 8000"
 