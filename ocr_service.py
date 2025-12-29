import numpy as np
import torch
import time
from PIL import Image

class OCRService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"OCRService initialized. Device: {self.device}")
        
        # Models cache
        self.models = {}
        self.current_model_name = None
        
        # Pre-load default (doctr) or lazy load
        # We will lazy load to save startup time
        
    def _load_doctr(self):
        if 'doctr' not in self.models:
            print("Loading Doctr recognition model...")
            from doctr.models import recognition_predictor
            # crnn_vgg16_bn is robust, crnn_mobilenet_v3 is faster
            self.models['doctr'] = recognition_predictor(arch='crnn_vgg16_bn', pretrained=True).to(self.device).eval()
    
    def _load_easyocr(self):
        if 'easyocr' not in self.models:
            print("Loading EasyOCR model...")
            import easyocr
            # Initialize for English by default, can be extended
            self.models['easyocr'] = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

    def _load_paddle(self):
        if 'paddle' not in self.models:
            print("Loading PaddleOCR model...")
            from paddleocr import PaddleOCR
            # use_angle_cls=True helps with rotated text
            # lang='en' by default
            self.models['paddle'] = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)

    def extract_text(self, image_input, text_regions, model_name='doctr'):
        """
        Extract text from provided regions in the image.
        Args:
            image_input: PIL Image or numpy array
            text_regions: List of dicts {'box': [x1, y1, x2, y2]}
            model_name: 'doctr' or 'easyocr'
        Returns:
            List of dicts: [{'box':..., 'text': "...", 'confidence': 0.9}]
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

        if model_name == 'doctr':
            self._load_doctr()
            # Doctr recognition_predictor expects list of numpy arrays
            # Output: list of objects with .value and .confidence
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
                    print(f"EasyOCR error on crop {i}: {e}")
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
                    # Current 'crop' is RGB (from PIL)
                    # Convert RGB -> BGR
                    crop_bgr = crop[..., ::-1]
                    
                    # Log shape for debugging (first run only maybe? or on error)
                    # print(f"[Debug] Paddle Input Shape: {crop_bgr.shape}")
                    
                    # Run ONLY classification (optional) and recognition. Disable detection.
                    # This is critical because we are feeding it tight crops from DBNet.
                    # Return list of (text, conf) tuples
                    result = ocr.ocr(crop_bgr, det=False, cls=True)
                    
                    # Full raw output log for debugging
                    # print(f"[Debug] Paddle Raw Result: {result}")
                    
                    full_text = ""
                    avg_conf = 0.0
                    
                    if result:
                        # With det=False, result is list of tuples: [('TEXT', 0.99), ...]
                        # Usually just one tuple for a single crop line
                        texts = [line[0] for line in result]
                        confs = [line[1] for line in result]
                        
                        full_text = " ".join(texts)
                        avg_conf = sum(confs) / len(confs)
                    # else:
                        # print(f"[Debug] Empty result from Paddle for crop {i}")
                    
                    results.append({
                        "box": valid_regions[i]['box'],
                        "text": full_text,
                        "confidence": float(avg_conf)
                    })
                    
                except Exception as e:
                    print(f"PaddleOCR error on crop {i}: {e}")
                    results.append({
                        "box": valid_regions[i]['box'],
                        "text": "",
                        "confidence": 0.0
                    })
        
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        t_inference = time.time() - t1
        
        print(f"[OCR Service] Model: {model_name} | Preprocess: {t_preprocess:.4f}s | Inference: {t_inference:.4f}s")
        
        return results, {
            "preprocess": t_preprocess,
            "inference": t_inference
        }
