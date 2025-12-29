import os
import torch
from PIL import Image
import numpy as np

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

class SAM3Service:
    def __init__(self, model_checkpoint_path="sam3/sam3.pt"):
        self.model = None
        self.processor = None
        self.model_path = model_checkpoint_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"SAM3 Service initialized. Device: {self.device}")


    def load_model(self):
        if self.model is not None:
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.model_path}. "
                "Please download it from Hugging Face: https://huggingface.co/facebook/sam3"
            )

        print("Loading SAM3 model... this may take a moment.")
        
        try:
            # Fix: Use correct argument 'checkpoint_path' and disable HF download
            # Also manually pass bpe_path to avoid NoneType error from pkg_resources
            
            # Calculate absolute path to BPE file
            # Assumes structure: project_root/sam3/sam3/assets/bpe...
            base_dir = os.path.dirname(os.path.abspath(__file__))
            bpe_path = os.path.join(base_dir, "sam3", "sam3", "assets", "bpe_simple_vocab_16e6.txt.gz")
            
            if not os.path.exists(bpe_path):
                 print(f"Warning: BPE path not found at {bpe_path}, trying default.")
                 bpe_path = None # Fallback to default logic if my guess is wrong
            else:
                 print(f"Using BPE path: {bpe_path}")

            self.model = build_sam3_image_model(
                checkpoint_path=self.model_path,
                device=self.device,
                load_from_HF=False,
                bpe_path=bpe_path
            )
            self.processor = Sam3Processor(self.model, device=self.device)
            print("SAM3 model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise e

    def detect(self, image: Image.Image, text_prompts: list[str]):
        """
        Run detection on an image for a list of text prompts.
        Returns a structured dictionary of results.
        """
        self.load_model()
        
        # SAM3 typically takes one prompt string. 
        # If we have multiple classes like ["cat", "dog"], we might need to join them 
        # or run them separately depending on how SAM3 handles "concepts".
        # The paper says "Segment Anything with Concepts".
        # Let's try sending them as a single string "cat. dog." or run loop.
        # For robustness and individual confidence control, running loop or 
        # if SAM3 supports list, using that is better. 
        # README example: prompt="<YOUR_TEXT_PROMPT>"
        
        results = []
        
        inference_state = self.processor.set_image(image)
        
        for class_name in text_prompts:
            # Run inference for this specific class concept
            output = self.processor.set_text_prompt(
                state=inference_state, 
                prompt=class_name
            )
            
            # Output keys: "masks", "boxes", "scores"
            # masks: [N, H, W] bool?
            # boxes: [N, 4]
            # scores: [N]
            
            masks = output["masks"]
            boxes = output["boxes"]
            scores = output["scores"]
            
            # Convert to friendly format
            # We'll rely on the caller to convert to JSON serializable
            
            # Check if tensors, move to cpu/numpy
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu().numpy().tolist()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy().tolist()
                
            # Masks are heavy to send as JSON. RLE (Run Length Encoding) or simplified polygons are better.
            # For this "Simple Platform", sending b64 encoded small mask 
            # or just bounding boxes first is easier.
            # But user wants specific visual feedback.
            # Let's convert masks to simple polygons or leave as is if handling backend rendering?
            # The plan says "frontend ... detection visualization". 
            # Typically easier to send [x,y,w,h] and a score to frontend for box.
            # For segmentation mask, generating a colored overlay on backend or sending polygon points.
            # Let's stick to Boxes + Scores + Counts first as per requirements for "Rapid".
            # "Rapid" focuses on detection counting.
            
            count = len(scores)
            
            class_result = {
                "class": class_name,
                "count": count,
                "detections": []
            }
            
            for i in range(count):
                det = {
                    "box": boxes[i], # [x1, y1, x2, y2] usually
                    "score": float(scores[i]),
                    # "mask": ... # Skip heavy mask data for initial JSON unless needed
                }
                class_result["detections"].append(det)
                
            results.append(class_result)
            
        return results

sam3_service = SAM3Service()
