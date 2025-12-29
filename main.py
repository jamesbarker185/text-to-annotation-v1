from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
from PIL import Image
import io
import json
import sys
import time
# Fix for sam3 shadowing: prioritized local sam3 package directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "sam3"))

from sam3_service import sam3_service
from dbnet_service import DBNetService
from ocr_service import OCRService

# Initialize Services
dbnet_service = DBNetService()
ocr_service = OCRService()

app = FastAPI(title="SAM3 Rapid Platform")

# CORS for development convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directory exists
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/api/detect")
async def detect_objects(
    file: UploadFile = File(...),
    prompts: str = Form(...)
):
    """
    Run SAM3 detection on a single uploaded image.
    prompts: comma separated string of class names e.g. "cat, dog, person"
    """
    try:
        # Read image
        t0 = time.time()
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Parse prompts
        prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
        
        if not prompt_list:
            raise HTTPException(status_code=400, detail="No prompt provided")
            
        # Run detection
        t_sam_start = time.time()
        results = sam3_service.detect(image, prompt_list)
        t_sam_end = time.time()
        sam3_duration = t_sam_end - t_sam_start
        
        # Run Text Detection (DBNet)
        t_db_start = time.time()
        text_regions = dbnet_service.detect_text(image)
        t_db_end = time.time()
        dbnet_duration = t_db_end - t_db_start
        
        t_total_end = time.time()
        total_duration = t_total_end - t0
        
        print(f"[Info] DBNet found {len(text_regions)} text regions.")
        print(f"\n[Performance] Detection Complete:")
        print(f"  - SAM3:  {sam3_duration:.4f}s")
        print(f"  - DBNet: {dbnet_duration:.4f}s")
        print(f"  - Total: {total_duration:.4f}s\n")
        
        # Save image temporarily for reference/serving if needed
        # (Optional, but good for debugging or returning URL)
        # temp_name = f"uploads/{file.filename}"
        # image.save(temp_name)
        
        # Get image dimensions for frontend scaling
        width, height = image.size
        
        return {
            "status": "success",
            "image_dims": {"width": width, "height": height},
            "results": results,
            "text_regions": text_regions,
            "timings": {
                "sam3": sam3_duration,
                "dbnet": dbnet_duration,
                "total": total_duration
            }
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-detect")
async def batch_detect(
    files: list[UploadFile] = File(...),
    prompts: str = Form(...),
    thresholds: str = Form(...) # JSON string: {"class_name": 0.5, ...}
):
    """
    Run detection on multiple images with fixed thresholds.
    Returns aggregated counts.
    """
    try:
        prompt_list = [p.strip() for p in prompts.split(",") if p.strip()]
        threshold_map = json.loads(thresholds)
        
        batch_results = []
        
        for file in files:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            
            # Run raw detection
            raw_results = sam3_service.detect(image, prompt_list)
            
            # Filter by threshold and count
            file_summary = {
                "filename": file.filename,
                "counts": {}
            }
            
            for res in raw_results:
                cls = res["class"]
                thresh = float(threshold_map.get(cls, 0.5)) # Default 0.5 if missing? Or 0.0?
                
                # Filter detections
                valid_dets = [d for d in res["detections"] if d["score"] >= thresh]
                
                file_summary["counts"][cls] = len(valid_dets)
            
            batch_results.append(file_summary)
            
        return {
            "status": "success",
            "batch_summary": batch_results
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/extract-text")
async def extract_text_api(
    file: UploadFile = File(...),
    regions: str = Form(...), # JSON string: [{"box": [x1, y1, x2, y2]}, ...]
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
        import traceback
        # traceback.print_exc()
        print(f"OCR Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("\n\n" + "="*50)
    print(" SAM3 Server is likely starting up...")
    print(" Once running, access the app at: http://localhost:8095")
    print("="*50 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8095, reload=True)
