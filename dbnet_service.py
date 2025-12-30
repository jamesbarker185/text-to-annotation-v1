
import numpy as np
import torch
import time
from doctr.models import detection_predictor
from PIL import Image
from threading import Lock

from config import get_settings
from logger import get_logger, log_performance

settings = get_settings()
logger = get_logger("dbnet_service")

class DBNetService:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DBNetService, cls).__new__(cls)
                    cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.DEVICE == "cuda" else "cpu")
        logger.info(f"DBNetService initialized. Device: {self.device}")
        
        self.model = None
        self.load_lock = Lock()
        self.initialized = True

    def ensure_model_loaded(self):
        if self.model is not None:
            return
            
        with self.load_lock:
            if self.model is not None:
                return
                
            logger.info("Loading DBNet model...")
            t0 = time.time()
            # Initialize pretrained DBNet (ResNet50 backbone)
            self.model = detection_predictor(arch='db_resnet50', pretrained=True).to(self.device).eval()
            log_performance(logger, "DBNet Model Load", time.time() - t0)

    def detect_text(self, image_input):
        """
        Detect text in an image.
        Args:
            image_input: PIL Image or numpy array
        """
        self.ensure_model_loaded()
        
        t0 = time.time()

        # Convert PIL to Numpy if needed
        if isinstance(image_input, Image.Image):
            img_np = np.array(image_input)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
        else:
            raise ValueError("Unsupported image format")

        try:
            # Doctr expects a list of numpy images
            with torch.no_grad():
                result = self.model([img_np])

            img_H, img_W = img_np.shape[:2]
            detections = []
            
            if len(result) > 0:
                prediction = result[0]
                # Based on previous analysis: key is 'words' and shape is (N, 5)
                # Format: relative coordinates [xmin, ymin, xmax, ymax, score]
                boxes_and_scores = prediction.get('words', np.empty((0, 5)))
                
                for item in boxes_and_scores:
                    if item.shape == (5,):
                        xmin, ymin, xmax, ymax, score = item
                        
                        # Convert to absolute
                        abs_box = [
                            int(xmin * img_W), 
                            int(ymin * img_H), 
                            int(xmax * img_W), 
                            int(ymax * img_H)
                        ]
                        
                        detections.append({
                            "box": abs_box,
                            "confidence": float(score)
                        })
            
            log_performance(logger, "DBNet Inference", time.time() - t0, {"detections": len(detections)})
            return detections
            
        except Exception as e:
            logger.error(f"DBNet detection error: {e}", exc_info=True)
            raise e
