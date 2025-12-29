import sys
import os

print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")

try:
    print("Attempting to import sam3...")
    import sam3
    print("Import sam3 successful")
    
    from sam3.model_builder import build_sam3_image_model
    print("Import build_sam3_image_model successful")
    
except Exception as e:
    print("\n!!! IMPORT ERROR !!!")
    import traceback
    traceback.print_exc()
