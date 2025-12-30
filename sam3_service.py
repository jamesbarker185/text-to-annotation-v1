
import os
import torch
from PIL import Image
import numpy as np
from threading import Lock

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from config import get_settings
from logger import get_logger, log_performance
import time

settings = get_settings()
logger = get_logger("sam3_service")

class SAM3Service:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SAM3Service, cls).__new__(cls)
                    cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        self.model = None
        self.processor = None
        self.model_path = settings.SAM3_CHECKPOINT
        # Fix: Prioritize 'cuda' if available, otherwise 'cpu'. 
        # The settings.DEVICE might be set to 'cuda', but if no cuda available, fallback.
        if settings.DEVICE == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        else:
            self.device = settings.DEVICE
            
        logger.info(f"SAM3 Service initialized. Target Device: {self.device}")
        self.initialized = True
        self.load_lock = Lock()

    def ensure_model_loaded(self):
        """Thread-safe model loading"""
        if self.model is not None:
            return

        with self.load_lock:
            if self.model is not None:
                return
                
            logger.info("Loading SAM3 model... this may take a moment.")
            t0 = time.time()
            self._load_model_internal()
            duration = time.time() - t0
            log_performance(logger, "SAM3 Model Load", duration)

    def _load_model_internal(self):
        if not os.path.exists(self.model_path):
            error_msg = (
                f"Model checkpoint not found at {self.model_path}. "
                "Please download it from Hugging Face: https://huggingface.co/facebook/sam3"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            # Calculate absolute path to BPE file
            base_dir = os.path.dirname(os.path.abspath(__file__))
            bpe_path = os.path.join(base_dir, "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
            
            if not os.path.exists(bpe_path):
                 logger.warning(f"BPE path not found at {bpe_path}, relying on default fallback.")
                 bpe_path = None

            self.model = build_sam3_image_model(
                checkpoint_path=self.model_path,
                device=self.device,
                load_from_HF=False,
                bpe_path=bpe_path
            )
            self.processor = Sam3Processor(self.model, device=self.device)
            logger.info("SAM3 model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            raise e

    def detect(self, image: Image.Image, text_prompts: list[str]):
        """
        Run detection on an image for a list of text prompts.
        """
        self.ensure_model_loaded()
        
        t0 = time.time()
        results = []
        
        try:
            inference_state = self.processor.set_image(image)
            
            for class_name in text_prompts:
                # Run inference for this specific class concept
                output = self.processor.set_text_prompt(
                    state=inference_state, 
                    prompt=class_name
                )
                
                masks = output["masks"]
                boxes = output["boxes"]
                scores = output["scores"]
                
                # Check if tensors, move to cpu/numpy
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy().tolist()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy().tolist()
                    
                count = len(scores)
                
                class_result = {
                    "class": class_name,
                    "count": count,
                    "detections": []
                }
                
                for i in range(count):
                    det = {
                        "box": boxes[i], # [x1, y1, x2, y2]
                        "score": float(scores[i]),
                    }
                    class_result["detections"].append(det)
                    
                results.append(class_result)
            
            log_performance(logger, "SAM3 Inference", time.time() - t0, {"prompts": len(text_prompts)})
            return results
            
        except Exception as e:
            logger.error(f"Error during SAM3 detection: {e}", exc_info=True)
            raise e

# Singleton instance
sam3_service = SAM3Service()
