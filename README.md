
# Text-to-Annotation V1

This project provides a rapid annotation backend using SAM3 for object detection and various OCR engines (DocTR, EasyOCR, PaddleOCR) for text extraction. It is designed to be scalable, performant, and easily deployable via Docker.

## Features
- **Object Detection**: Segment Anything Model 3 (SAM3) integration.
- **Text Detection**: DBNet (via DocTR) for robust text localization.
- **text Recognition**: Pluggable OCR backends (DocTR, EasyOCR, PaddleOCR).
- **Architecture**: Singleton service pattern, lazy loading, and thread-safe execution.
- **Deployment**: Dockerized with health checks and GPU support.

## Getting Started

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU (Recommended) + NVIDIA Container Toolkit

### Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure `.env`:
   ```env
   API_PORT=8095
   DEVICE=cuda
   ```
3. Run the server:
   ```bash
   python main.py
   ```
   Access at `http://localhost:8095`

### Docker Deployment
1. Build and run:
   ```bash
   docker-compose up --build -d
   ```
2. Check health:
   ```bash
   curl http://localhost:8095/api/health
   ```

## Configuration
All configuration is managed via `config.py` and environment variables. Key variables include:
- `API_PORT`: Port to listen on.
- `DEVICE`: `cuda` or `cpu`.
- `LOG_LEVEL`: Logging verbosity (INFO, DEBUG, etc.).

## Project Structure
- `main.py`: FastAPI entry point and logic.
- `sam3_service.py`: Wrapper for SAM3 model.
- `ocr_service.py`: Wrapper for OCR engines.
- `dbnet_service.py`: Wrapper for DBNet text detection.
- `config.py`: Centralized settings.
- `logger.py`: Structured logging configuration.
- `static/`: Lightweight frontend for testing.
