import sys
import os
import traceback

print("Checking Sam3Processor dependencies...")
try:
    from sam3.model.sam3_image_processor import Sam3Processor
    print("SUCCESS: Sam3Processor imported.")
except Exception as e:
    print("\n!!! ERROR IMPORTING Sam3Processor !!!")
    traceback.print_exc()
