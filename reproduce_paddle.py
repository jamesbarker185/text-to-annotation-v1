
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

# Use one of the uploaded images if possible, or create a dummy "text-like" image
# Since I can't access user's uploads directly repeatedly, I will create a synthetic image that looks like a crop

def create_dummy_crop():
    # Create a white image
    img = np.ones((64, 200, 3), dtype=np.uint8) * 255
    # Add text "TESTING"
    cv2.putText(img, "PADDLE", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    return img

if __name__ == "__main__":
    ocr = PaddleOCR(use_angle_cls=True, lang='en', enable_mkldnn=False)
    
    crop = create_dummy_crop()
    # Ensure BGR (OpenCV default is BGR, but checking)
    
    print("-" * 30)
    print("Test 1: det=True (Default)")
    res_det = ocr.ocr(crop, det=True, cls=True)
    print(f"Result: {res_det}")
    
    print("-" * 30)
    print("Test 2: det=False (Rec Only)")
    try:
        res_rec = ocr.ocr(crop, det=False, cls=True)
        print(f"Result: {res_rec}")
    except Exception as e:
        print(f"Error: {e}")
