# VehicleBERT Deployment Guide

Guide for deploying VehicleBERT in production environments.

## Table of Contents

1. [FastAPI REST API](#fastapi-rest-api)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Model Optimization](#model-optimization)
5. [Monitoring](#monitoring)

## FastAPI REST API

### Create API Server

Create `api/app.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import sys
sys.path.insert(0, '../src')

from vehiclebert import VehicleBERTPredictor

# Initialize FastAPI
app = FastAPI(
    title="VehicleBERT API",
    description="Named Entity Recognition for Automotive Text",
    version="1.0.0"
)

# Load model at startup
predictor = None

@app.on_event("startup")
async def load_model():
    global predictor
    predictor = VehicleBERTPredictor.from_pretrained("../models/vehiclebert/best")

# Request/Response models
class TextRequest(BaseModel):
    text: str
    confidence_threshold: float = 0.5

class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int
    confidence: float

class PredictionResponse(BaseModel):
    text: str
    entities: List[Entity]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    """Predict entities in text"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        entities = predictor.predict(
            request.text,
            return_confidence=True
        )
        
        # Filter by confidence
        entities = [
            e for e in entities 
            if e['confidence'] >= request.confidence_threshold
        ]
        
        return PredictionResponse(
            text=request.text,
            entities=entities
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict(texts: List[str], batch_size: int = 8):
    """Predict entities for multiple texts"""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = predictor.predict_batch(texts, batch_size=batch_size)
        return {
            "predictions": [
                {"text": text, "entities": entities}
                for text, entities in zip(texts, results)
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }

@app.get("/info")
async def info():
    """Get model information"""
    return {
        "model": "VehicleBERT",
        "version": "1.0.0",
        "entity_types": [
            "VEHICLE_PART", "DIAGNOSTIC_CODE", "SENSOR", "ECU",
            "PROTOCOL", "SYMPTOM", "TOOL", "MEASUREMENT",
            "PROCEDURE", "FLUID", "MANUFACTURER", "MODEL"
        ]
    }
```

### Run API Server

```bash
# Install FastAPI and uvicorn
pip install fastapi uvicorn

# Run server
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Test API

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Replace the O2 sensor and clear P0420 from ECM"}'
```

## Docker Deployment

### Create Dockerfile

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY models/ ./models/
COPY api/ ./api/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Create docker-compose.yml

```yaml
version: '3.8'

services:
  vehiclebert:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/vehiclebert/best
    volumes:
      - ./models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Build and Run

```bash
# Build image
docker build -t vehiclebert:latest .

# Run container
docker run -p 8000:8000 vehiclebert:latest

# Or use docker-compose
docker-compose up -d
```

## Cloud Deployment

### AWS Deployment (EC2)

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04
   - Instance type: t3.medium (minimum)
   - Storage: 20GB+

2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install -y python3-pip git
   ```

3. **Clone and Setup**
   ```bash
   git clone https://github.com/yourusername/vehiclebert.git
   cd vehiclebert
   pip3 install -r requirements.txt
   ```

4. **Run with systemd**
   
   Create `/etc/systemd/system/vehiclebert.service`:
   ```ini
   [Unit]
   Description=VehicleBERT API
   After=network.target

   [Service]
   Type=simple
   User=ubuntu
   WorkingDirectory=/home/ubuntu/vehiclebert
   ExecStart=/usr/bin/python3 -m uvicorn api.app:app --host 0.0.0.0 --port 8000
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable vehiclebert
   sudo systemctl start vehiclebert
   ```

### Google Cloud Run

1. **Create Dockerfile** (see Docker section above)

2. **Build and Push**
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/vehiclebert
   ```

3. **Deploy**
   ```bash
   gcloud run deploy vehiclebert \
     --image gcr.io/PROJECT_ID/vehiclebert \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

### Azure Container Instances

```bash
# Build and push to Azure Container Registry
az acr build --registry myregistry --image vehiclebert:v1 .

# Deploy
az container create \
  --resource-group myResourceGroup \
  --name vehiclebert \
  --image myregistry.azurecr.io/vehiclebert:v1 \
  --dns-name-label vehiclebert \
  --ports 8000
```

## Model Optimization

### 1. Model Quantization

```python
import torch
from vehiclebert import VehicleBERTModel

# Load model
model = VehicleBERTModel.load("models/vehiclebert/best")

# Quantize to int8
quantized_model = torch.quantization.quantize_dynamic(
    model.model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Save quantized model
model.model = quantized_model
model.save("models/vehiclebert/quantized")
```

### 2. ONNX Export

```python
import torch
from vehiclebert import VehicleBERTModel

model = VehicleBERTModel.load("models/vehiclebert/best")
model.model.eval()

# Dummy input
dummy_input = torch.randint(0, 1000, (1, 128))
dummy_mask = torch.ones((1, 128))

# Export to ONNX
torch.onnx.export(
    model.model,
    (dummy_input, dummy_mask),
    "models/vehiclebert/model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['output'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'output': {0: 'batch_size', 1: 'sequence'}
    }
)
```

### 3. TensorRT Optimization

For NVIDIA GPUs:

```python
# Convert ONNX to TensorRT
import tensorrt as trt

# Build TensorRT engine
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

# Parse ONNX model
with open("models/vehiclebert/model.onnx", "rb") as f:
    parser.parse(f.read())

# Build engine
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
engine = builder.build_engine(network, config)
```

## Monitoring

### Prometheus Metrics

Add to `api/app.py`:

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')

@app.post("/predict")
@prediction_duration.time()
async def predict(request: TextRequest):
    prediction_counter.inc()
    # ... existing code ...

@app.get("/metrics")
async def metrics():
    return generate_latest()
```

### Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('vehiclebert.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(request: TextRequest):
    logger.info(f"Prediction request: {request.text[:50]}...")
    # ... existing code ...
```

## Load Balancing

### Nginx Configuration

```nginx
upstream vehiclebert {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://vehiclebert;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Security

### API Key Authentication

```python
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = "your-secret-key"
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

@app.post("/predict", dependencies=[Security(verify_api_key)])
async def predict(request: TextRequest):
    # ... existing code ...
```

## Performance Benchmarks

Expected performance on different hardware:

| Hardware | Batch Size | Throughput | Latency |
|----------|------------|------------|---------|
| CPU (8 cores) | 1 | ~20 req/s | ~50ms |
| CPU (8 cores) | 8 | ~60 req/s | ~130ms |
| GPU (T4) | 1 | ~100 req/s | ~10ms |
| GPU (T4) | 32 | ~500 req/s | ~64ms |

---

For questions or issues, please open an issue on GitHub!
