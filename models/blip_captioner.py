import torch
from PIL import Image
from typing import List, Dict, Optional, Tuple
from transformers import BlipProcessor, BlipForConditionalGeneration


class BLIPSceneCaptioner:
    """
    Integrates the Bootstrapped Language-Image Pre-training (BLIP) foundation model
    as specified in Impraimakis et al. (arXiv:2603.23037).
    
    Generates natural language scene descriptions and object-level semantic context
    to create a complete multimodal chain of interpretability alongside the KAN surrogate.
    """
    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-base",
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[BLIP] Loading vision-language foundation model '{model_name}' on {self.device}...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print("[BLIP] Model loaded successfully.")

    def caption_image(
        self,
        image: Image.Image,
        prompt: Optional[str] = None,
        max_new_tokens: int = 50,
    ) -> str:
        """
        Generates a natural language caption describing the scene.
        Args:
            image: PIL Image object
            prompt: Optional text prompt (conditional captioning)
            max_new_tokens: Maximum tokens in generated description
        Returns:
            caption: Natural language string describing the visual scene
        """
        if prompt:
            inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
        return caption

    def generate_multimodal_report(
        self,
        image: Image.Image,
        detections: List[Dict[str, any]],
        trust_threshold: float = 0.5,
    ) -> Dict[str, any]:
        """
        Builds a comprehensive multimodal audit combining:
          1. BLIP natural language scene understanding
          2. YOLOv10 object detections
          3. KAN 7-feature trustworthiness estimates & degradation flags
        """
        scene_caption = self.caption_image(image)

        verified_detections = []
        low_trust_detections = []

        for det in detections:
            conf = det.get("confidence", 0.0)
            trust = det.get("trust_score", conf)
            cls_name = det.get("class_name", "object")

            det_summary = {
                "class_name": cls_name,
                "box_xyxy": det.get("box_xyxy"),
                "yolo_conf": round(float(conf), 3),
                "kan_trust": round(float(trust), 3),
                "is_trustworthy": bool(trust >= trust_threshold),
            }

            # Flag discrepancy between high raw confidence and low trust score
            if conf >= 0.5 and trust < trust_threshold:
                det_summary["warning"] = "High YOLO confidence but low KAN trust (possible occlusion/blur)"
                low_trust_detections.append(det_summary)
            else:
                verified_detections.append(det_summary)

        return {
            "blip_scene_caption": scene_caption,
            "total_detections": len(detections),
            "verified_count": len(verified_detections),
            "low_trust_count": len(low_trust_detections),
            "verified_detections": verified_detections,
            "low_trust_detections": low_trust_detections,
        }
