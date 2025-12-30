
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import io
import json
import sys
import time
from PIL import Image

# Infrastructure
from config import get_settings
from logger import get_logger, log_performance
from middleware import performance_logging_middleware

# Initialize Infrastructure
settings = get_settings()
logger = get_logger("api")

# Ensure SAM3 is in path
sys.path.insert(0, os.path.join(settings.BASE_DIR, "sam3"))

# Services (Singletons)
from sam3_service import sam3_service
from dbnet_service import DBNetService
from ocr_service import OCRService

# Initialize Service Instances
dbnet_service = DBNetService()
ocr_service = OCRService()

app = FastAPI(title=settings.API_TITLE, version=settings.API_VERSION)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Configure in .env for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(performance_logging_middleware)

# Ensure directories exist
os.makedirs(settings.STATIC_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

@app.get("/")
async def read_root():
    return FileResponse(os.path.join(settings.STATIC_DIR, "index.html"))

@app.get("/health")
async def health_check():
    """Basic container health check"""
    return {"status": "ok"}

@app.get("/api/health")
async def detailed_health():
    """Detailed service health"""
    return {
        "status": "ok",
        "services": {
            "sam3": "initialized" if sam3_service.initialized else "pending",
            "dbnet": "initialized" if dbnet_service.initialized else "pending",
            "ocr": "initialized" if ocr_service.initialized else "pending"
        },
        "config": {
            "device": settings.DEVICE,
            "lazy_load": settings.LAZY_LOAD_MODELS
        }
    }

@app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...), prompts: str = Form(...)):
    """
    Run SAM3 detection on a single uploaded image.
    """
    try:
        t0 = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
        if not prompt_list:
            raise HTTPException(status_code=400, detail="No prompt provided")
            
        # Run SAM3
        t_sam_start = time.time()
        results = sam3_service.detect(image, prompt_list)
        t_sam_end = time.time()
        
        # Run DBNet
        t_db_start = time.time()
        text_regions = dbnet_service.detect_text(image)
        t_db_end = time.time()
        
        total_duration = time.time() - t0
        
        width, height = image.size
        
        return {
            "status": "success",
            "image_dims": {"width": width, "height": height},
            "results": results,
            "text_regions": text_regions,
            "timings": {
                "sam3": t_sam_end - t_sam_start,
                "dbnet": t_db_end - t_db_start,
                "total": total_duration
            }
        }
        
    except Exception as e:
        logger.error(f"Detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-detect")
async def batch_detect(
    files: list[UploadFile] = File(...),
    prompts: str = Form(...),
    thresholds: str = Form(...) 
):
    """
    Run detection on multiple images.
    """
    try:
        prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
        threshold_map = json.loads(thresholds)
        batch_results = []
        
        t0 = time.time()
        
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            raw_results = sam3_service.detect(image, prompt_list)
            
            file_summary = {
                "filename": file.filename,
                "counts": {}
            }
            
            for res in raw_results:
                cls = res["class"]
                thresh = float(threshold_map.get(cls, 0.5))
                valid_dets = [d for d in res["detections"] if d["score"] >= thresh]
                file_summary["counts"][cls] = len(valid_dets)
            
            batch_results.append(file_summary)
            
        log_performance(logger, "Batch Detection", time.time() - t0, {"files": len(files)})
            
        return {
            "status": "success",
            "batch_summary": batch_results
        }
    except Exception as e:
        logger.error(f"Batch detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract-text")
async def extract_text_api(
    file: UploadFile = File(...),
    regions: str = Form(...), 
    model: str = Form("doctr")
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        region_list = json.loads(regions)
        
        extracted_data, perf_stats = ocr_service.extract_text(image, region_list, model_name=model)
        
        return {
            "status": "success",
            "extracted_text": extracted_data,
            "perf_stats": perf_stats
        }
    except Exception as e:
        logger.error(f"Text extraction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("="*50)
    logger.info(f"Starting {settings.API_TITLE} on port {settings.API_PORT}")
    logger.info(f"Access at: http://{settings.API_HOST}:{settings.API_PORT}")
    logger.info("="*50)
    
    uvicorn.run(
        "main:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT, 
        reload=settings.DEBUG
    )
