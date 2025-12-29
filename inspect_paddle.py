
from paddleocr import PaddleOCR
import inspect

ocr = PaddleOCR(use_angle_cls=True, lang="en")
print(f"PaddleOCR.ocr signature: {inspect.signature(ocr.ocr)}")
print(f"PaddleOCR dir: {dir(ocr)}")
