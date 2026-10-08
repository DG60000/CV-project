import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
from ultralytics import YOLO
from .kan_surrogate import extract_7_features


class YOLOv10Detector:
    """
    Standard YOLOv10 perception module as specified in Impraimakis et al. (arXiv:2603.23037).
    Executes real-time object detection and formats raw bounding box detections
    into the 7 geometric and semantic features needed by the KAN surrogate model.
    """
    def __init__(
        self,
        model_path: str = "yolov10n.pt",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[YOLOv10] Loading model '{model_path}' on device '{self.device}'...")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_names = self.model.names

    def detect(
        self,
        image: Image.Image,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
    ) -> List[Dict[str, any]]:
        """
        Runs object detection on a PIL Image.
        Returns a list of structured detection dictionaries including the 7 KAN features.
        """
        conf_thresh = conf if conf is not None else self.conf_threshold
        iou_thresh = iou if iou is not None else self.iou_threshold

        results = self.model(image, conf=conf_thresh, iou=iou_thresh, device=self.device, verbose=False)[0]
        orig_w, orig_h = image.size

        detections = []
        boxes_data = results.boxes

        if boxes_data is None or len(boxes_data) == 0:
            return detections

        # Extract coordinates
        xyxy = boxes_data.xyxy.cpu().numpy()  # [N, 4]
        confidences = boxes_data.conf.cpu().numpy()  # [N]
        class_ids = boxes_data.cls.cpu().numpy().astype(int)  # [N]

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            c = float(confidences[i])
            cls_id = int(class_ids[i])
            cls_name = self.class_names.get(cls_id, f"class_{cls_id}")

            # Normalized geometry (center-x, center-y, width, height) in [0, 1]
            box_w = (x2 - x1) / orig_w
            box_h = (y2 - y1) / orig_h
            cx = (x1 + x2) / (2.0 * orig_w)
            cy = (y1 + y2) / (2.0 * orig_h)

            # Clamp normalized values to [0, 1]
            cx = max(0.0, min(1.0, float(cx)))
            cy = max(0.0, min(1.0, float(cy)))
            box_w = max(0.0, min(1.0, float(box_w)))
            box_h = max(0.0, min(1.0, float(box_h)))

            # Relative scale: (w_px * h_px) / 640^2
            scale = float(box_w * box_h)

            # Exactly the 7 features
            feat_vector = np.array([cx, cy, box_w, box_h, c, cls_id / 80.0, scale], dtype=np.float32)

            detections.append({
                "box_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "box_norm_xywh": [cx, cy, box_w, box_h],
                "confidence": c,
                "class_id": cls_id,
                "class_name": cls_name,
                "relative_scale": scale,
                "features_7": feat_vector,
            })

        return detections
