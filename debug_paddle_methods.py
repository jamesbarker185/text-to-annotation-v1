
from paddleocr import PaddleOCR
import traceback
import sys

try:
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    print("Methods:", dir(ocr))
    print("Help ocr.ocr:")
    help(ocr.ocr)
    
    import numpy as np
    img = np.zeros((100,100,3), dtype=np.uint8)
    
    print("Running ocr(det=False)...")
    res = ocr.ocr(img, det=False, cls=True)
    print("Result:", res)

except Exception:
    with open("paddle_error.txt", "w") as f:
        traceback.print_exc(file=f)
    traceback.print_exc()
