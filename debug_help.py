
from paddleocr import PaddleOCR
import sys

ocr = PaddleOCR(use_angle_cls=True, lang='en')

with open("paddle_help.txt", "w", encoding='utf-8') as f:
    f.write(f"Version info: {ocr.__class__}\n")
    f.write(f"Dir(ocr): {dir(ocr)}\n")
    
    f.write("\nHelp ocr.ocr:\n")
    sys.stdout = f
    help(ocr.ocr)
    
    if hasattr(ocr, 'predict'):
        f.write("\nHelp ocr.predict:\n")
        help(ocr.predict)
