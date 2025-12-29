
import paddleocr
import inspect

print(f"paddleocr file: {paddleocr.__file__}")

try:
    from paddleocr.paddleocr import PaddleOCR as LegacyPaddleOCR
    print("Found paddleocr.paddleocr.PaddleOCR")
    ocr = LegacyPaddleOCR(use_angle_cls=True, lang='en')
    print(f"Legacy signature: {inspect.signature(ocr.ocr)}")
except ImportError:
    print("Could not import paddleocr.paddleocr.PaddleOCR")
except Exception as e:
    print(f"Error init legacy: {e}")
