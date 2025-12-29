
import numpy as np
import torch
from doctr.models import detection_predictor
from doctr.io import DocumentFile
from PIL import Image

class DBNetService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"DBNetService initialized. Device: {self.device}")
        # Initialize pretrained DBNet (ResNet50 backbone)
        self.model = detection_predictor(arch='db_resnet50', pretrained=True).to(self.device).eval()

    def detect_text(self, image_input):
        """
        Detect text in an image.
        Args:
            image_input: PIL Image or numpy array
        Returns:
            List of text regions. Each region is a dictionary with:
            - box: [x1, y1, x2, y2] (absolute coordinates)
            - confidence: score (float)
        """
        # Convert PIL to Numpy if needed
        if isinstance(image_input, Image.Image):
            img_np = np.array(image_input)
        elif isinstance(image_input, np.ndarray):
            img_np = image_input
        else:
            raise ValueError("Unsupported image format")

        # Doctr expects a list of numpy images
        # The predictor handles resizing internally
        with torch.no_grad():
            result = self.model([img_np])

        # Result is a list of maps (one per page/image)
        # result[0] is the dictionary for the single image containing 'boxes' (and maybe 'classes')
        # Actually, for detection_predictor, result is a list of maps, where each map is a dict 
        # or it returns just the boxes directly depending on post-processing.
        # Let's standardize based on test_doctr.py output (once confirmed), 
        # but typically detection_predictor returns: [ {'words': array(N, 4, 2), ...} ]?
        # NO, doctr.models.detection.predictor returns a standard output format.
        # result is a list of numpy arrays (N, 5) or (N, 4, 2) ?
        
        # Based on docs: result is a list of (N, 5) for db_resnet50? 
        # Actually it returns absolute/relative coordinates.
        # Let's robustly check the shape in usage. 
        # Usually it returns relative coordinates [xmin, ymin, xmax, ymax, score] or polygon.

        # For now, we assume it returns dictionary or list of boxes.
        # Inspecting source: detection_predictor returns a DetectionPredictor which returns a list of maps.
        # Each map is a dictionary containing 'boxes' (N, 4, 2) relative coordinates? 
        # Or keys: 'boxes', 'scores'.
        
        # Wait, DetectionPredictor.__call__ returns list of maps (N, 4, 2) if assumes_straight_pages=True?
        # Let's rely on the result being a list (one per image).
        
        img_H, img_W = img_np.shape[:2]
        
        detections = []
        
        if len(result) > 0:
            prediction = result[0]
            # Based on test_doctr.py output, key is 'words' and shape is (N, 5)
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
            
        return detections
