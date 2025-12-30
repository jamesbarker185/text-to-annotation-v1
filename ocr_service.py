
import numpy as np
import torch
import time
from PIL import Image
from threading import Lock
from functools import lru_cache

from config import get_settings
from logger import get_logger, log_performance

settings = get_settings()
logger = get_logger("ocr_service")

class OCRService:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(OCRService, cls).__new__(cls)
                    cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        self.device = torch.device("cuda" if torch.cuda.is_available() and settings.DEVICE == "cuda" else "cpu")
        logger.info(f"OCRService initialized. Device: {self.device}")
        
        # Models cache
        self.models = {}
        self.model_locks = {
            'doctr': Lock(),
            'easyocr': Lock(),
            'paddle': Lock()
        }
        self.initialized = True
        
    def _load_doctr(self):
        if 'doctr' in self.models:
            return

        with self.model_locks['doctr']:
            if 'doctr' in self.models:
                return
                
            logger.info("Loading Doctr model...")
            t0 = time.time()
            from doctr.models import ocr_predictor
            # Pretrained defaults to True
            self.models['doctr'] = ocr_predictor(pretrained=True).reco_predictor.to(self.device).eval()
            log_performance(logger, "Doctr Load", time.time() - t0)
    
    def _load_easyocr(self):
        if 'easyocr' in self.models:
            return

        with self.model_locks['easyocr']:
            if 'easyocr' in self.models:
                return

            logger.info("Loading EasyOCR model...")
            t0 = time.time()
            import easyocr
            self.models['easyocr'] = easyocr.Reader(['en'], gpu=(self.device.type == 'cuda'))
            log_performance(logger, "EasyOCR Load", time.time() - t0)

    def _load_paddle(self):
        if 'paddle' in self.models:
            return

        with self.model_locks['paddle']:
            if 'paddle' in self.models:
                return

            logger.info("Loading PaddleOCR model...")
            t0 = time.time()
            
            # Explicitly disable MKLDNN via environment variables and flags
            import os
            os.environ["FLAGS_use_mkldnn"] = "0"
            os.environ["DN_ENABLE_ONEDNN"] = "0"

            try:
                import paddle
                paddle.set_flags({'FLAGS_use_mkldnn': False})
                logger.info(f"Paddle flags: {paddle.get_flags(['FLAGS_use_mkldnn'])}")
            except ImportError:
                logger.warning("Could not import paddle to set flags")

            from paddleocr import PaddleOCR
            # use_angle_cls=True helps with rotated text
            # lang='en' by default
            # Paddle uses its own GPU check usually, but we can hint
            use_gpu = (self.device.type == 'cuda')
            logger.info(f"Initializing PaddleOCR with use_gpu={use_gpu}, enable_mkldnn=False, det=False, rec_batch_num=1")
            self.models['paddle'] = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=use_gpu, enable_mkldnn=False, det=False, rec_batch_num=1)
            log_performance(logger, "PaddleOCR Load", time.time() - t0)

    def extract_text(self, image_input, text_regions, model_name='doctr'):
        """
        Extract text from provided regions in the image.
        """
        if not text_regions:
            return [], {}

        t0 = time.time()

        # Convert PIL to Numpy if needed
        if isinstance(image_input, Image.Image):
            img_np = np.array(image_input)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
        else:
            raise ValueError("Unsupported image format")

        results = []
        
        # Crop images
        crops = []
        valid_regions = []
        
        H, W = img_np.shape[:2]
        
        for region in text_regions:
            x1, y1, x2, y2 = region['box']
            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop = img_np[y1:y2, x1:x2]
            crops.append(crop)
            valid_regions.append(region)

        t_preprocess = time.time() - t0
        t1 = time.time()

        if not crops:
            return [], {}

        try:
            if model_name == 'doctr':
                self._load_doctr()
                # Doctr recognition_predictor expects list of numpy arrays
                out = self.models['doctr'](crops)
                
                for i, word_out in enumerate(out):
                    text, confidence = word_out[0], word_out[1]
                    results.append({
                        "box": valid_regions[i]['box'],
                        "text": text,
                        "confidence": float(confidence)
                    })
                    
            elif model_name == 'easyocr':
                self._load_easyocr()
                reader = self.models['easyocr']
                
                for i, crop in enumerate(crops):
                    try:
                        ocr_res = reader.readtext(crop, detail=1)
                        full_text = " ".join([res[1] for res in ocr_res])
                        avg_conf = 0.0
                        if ocr_res:
                            avg_conf = sum([res[2] for res in ocr_res]) / len(ocr_res)
                        
                        results.append({
                            "box": valid_regions[i]['box'],
                            "text": full_text,
                            "confidence": float(avg_conf)
                        })
                    except Exception as e:
                        logger.warning(f"EasyOCR error on crop {i}: {e}")
                        results.append({
                            "box": valid_regions[i]['box'],
                            "text": "",
                            "confidence": 0.0
                        })
            
            elif model_name == 'paddle':
                self._load_paddle()
                ocr = self.models['paddle']
                
                for i, crop in enumerate(crops):
                    try:
                        # PaddleOCR expects BGR format for numpy arrays
                        crop_bgr = crop[..., ::-1]
                        
                        # Run ONLY classification and recognition
                        result = ocr.ocr(crop_bgr, det=False, cls=True)
                        
                        full_text = ""
                        avg_conf = 0.0
                        
                        if result:
                            # result format is usually [[('Text', 0.99), ...]]
                            try:
                                # Flatten if list of lists
                                if isinstance(result[0], list):
                                    line_res = result[0]
                                else:
                                    line_res = result # Just in case structure varies
                                
                                # Safe extraction
                                texts = []
                                confs = []
                                pass_items = line_res if line_res else []
                                for item in pass_items:
                                    if item is not None and len(item) >= 2:
                                        texts.append(item[0])
                                        confs.append(item[1])
                                
                                full_text = " ".join(texts)
                                if confs:
                                    avg_conf = sum(confs) / len(confs)
                                        
                            except Exception as e:
                                 logger.warning(f"PaddleOCR parsing warning: {e}. Raw: {result}")
                        
                        results.append({
                            "box": valid_regions[i]['box'],
                            "text": full_text,
                            "confidence": float(avg_conf)
                        })
                        
                    except Exception as e:
                        logger.warning(f"PaddleOCR error on crop {i}: {e}")
                        results.append({
                            "box": valid_regions[i]['box'],
                            "text": "",
                            "confidence": 0.0
                        })
            
            else:
                raise ValueError(f"Unknown model name: {model_name}")

        except Exception as e:
            logger.error(f"OCR Inference Error ({model_name}): {e}", exc_info=True)
            raise e

        t_inference = time.time() - t1
        
        log_performance(logger, f"OCR ({model_name})", t_inference, {
            "crops": len(crops),
            "preprocess": f"{t_preprocess:.4f}s"
        })
        
        return results, {
            "preprocess": t_preprocess,
            "inference": t_inference
        }
