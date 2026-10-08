import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def letterbox(img: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


class COCODetectionDataset(Dataset):
    """
    Loads COCO images and YOLO-formatted label files (.txt).
    Label format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1).
    """
    def __init__(
        self,
        img_dir: str,
        label_dir: str,
        img_size: int = 640,
        num_classes: int = 80,
        strides: List[int] = [8, 16, 32],
        reg_max: int = 16,
    ):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max
        self.coco_json_annotations = None

        # Check if label_dir is a JSON file or directory containing COCO JSON
        json_path = None
        if os.path.isfile(label_dir) and label_dir.endswith(".json"):
            json_path = label_dir
        elif os.path.isdir(label_dir):
            for f in os.listdir(label_dir):
                if f.startswith("instances_") and f.endswith(".json"):
                    json_path = os.path.join(label_dir, f)
                    break

        if json_path and os.path.exists(json_path):
            import json
            print(f"[Dataset] Parsing COCO JSON annotations from {json_path}...")
            with open(json_path, "r") as f:
                cdata = json.load(f)
            id_to_fname = {img["id"]: img["file_name"] for img in cdata["images"]}
            self.coco_json_annotations = {}
            for ann in cdata["annotations"]:
                fname = id_to_fname.get(ann["image_id"])
                if fname:
                    if fname not in self.coco_json_annotations:
                        self.coco_json_annotations[fname] = []
                    bx, by, bw, bh = ann["bbox"]
                    self.coco_json_annotations[fname].append([bx, by, bw, bh, ann["category_id"]])
            print(f"[Dataset] Indexed {len(self.coco_json_annotations)} image annotations from JSON.")

        # Find all matching image paths
        valid_extensions = ("*.jpg", "*.jpeg", "*.png")
        self.img_files = []
        for ext in valid_extensions:
            self.img_files.extend(glob.glob(os.path.join(img_dir, ext)))
        self.img_files.sort()

        if len(self.img_files) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}. Check your COCO dataset path.")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx: int):
        img_path = self.img_files[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to read image at {img_path}")

        orig_h, orig_w = img.shape[:2]
        img_padded, ratio, (dw, dh) = letterbox(img, new_shape=(self.img_size, self.img_size))

        # Convert BGR -> RGB and normalize to [0, 1]
        img_tensor = torch.from_numpy(img_padded).permute(2, 0, 1).float() / 255.0

        # Load corresponding label file
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        full_fname = os.path.basename(img_path)
        label_path = os.path.join(self.label_dir, f"{base_name}.txt")

        boxes = []
        class_ids = []

        if self.coco_json_annotations and full_fname in self.coco_json_annotations:
            for item in self.coco_json_annotations[full_fname]:
                bx, by, bw, bh, cat_id = item
                # Convert COCO xywh pixel coords to padded pixel coordinates
                xc_px = (bx + bw / 2.0) * ratio + dw
                yc_px = (by + bh / 2.0) * ratio + dh
                w_px = bw * ratio
                h_px = bh * ratio
                boxes.append([xc_px, yc_px, w_px, h_px])
                class_ids.append(min(cat_id, self.num_classes - 1))
        elif os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        # Normalized coordinates
                        xc, yc, w, h = map(float, parts[1:5])
                        
                        # Convert normalized original coords to padded pixel coordinates
                        xc_px = (xc * orig_w * ratio) + dw
                        yc_px = (yc * orig_h * ratio) + dh
                        w_px = w * orig_w * ratio
                        h_px = h * orig_h * ratio

                        boxes.append([xc_px, yc_px, w_px, h_px])
                        class_ids.append(cls_id)

        # Build Multi-scale Ground Truth Targets (P3: 80x80, P4: 40x40, P5: 20x20)
        grid_sizes = [self.img_size // s for s in self.strides]
        cls_targets = [torch.zeros(self.num_classes, gs, gs) for gs in grid_sizes]

        pos_pred_boxes = []
        pos_target_boxes = []
        pos_pred_dfl = []
        pos_target_dfl = []

        for box, cls_id in zip(boxes, class_ids):
            xc, yc, bw, bh = box
            for scale_idx, stride in enumerate(self.strides):
                gx, gy = int(xc // stride), int(yc // stride)
                gs = grid_sizes[scale_idx]
                if 0 <= gx < gs and 0 <= gy < gs:
                    cls_targets[scale_idx][cls_id, gy, gx] = 1.0

            # Store positive bounding box target for CIoU / DFL loss computation
            norm_box = torch.tensor([xc / self.img_size, yc / self.img_size, bw / self.img_size, bh / self.img_size])
            pos_target_boxes.append(norm_box)
            pos_pred_boxes.append(norm_box)  # Training target anchor reference

            # Target DFL boundary distances (left, top, right, bottom)
            dfl_dist = torch.tensor([
                (xc - bw / 2) / self.strides[0],
                (yc - bh / 2) / self.strides[0],
                (self.img_size - (xc + bw / 2)) / self.strides[0],
                (self.img_size - (yc + bh / 2)) / self.strides[0],
            ]).clamp(0, self.reg_max - 1.01)

            pos_target_dfl.append(dfl_dist)
            pos_pred_dfl.append(torch.randn(4, self.reg_max))

        if len(pos_target_boxes) > 0:
            target_boxes_tensor = torch.stack(pos_target_boxes, dim=0)
            pred_boxes_tensor = torch.stack(pos_pred_boxes, dim=0)
            target_dfl_tensor = torch.stack(pos_target_dfl, dim=0)
            pred_dfl_tensor = torch.stack(pos_pred_dfl, dim=0)
            pos_mask = torch.ones(len(pos_target_boxes), dtype=torch.bool)
        else:
            target_boxes_tensor = torch.empty((0, 4))
            pred_boxes_tensor = torch.empty((0, 4))
            target_dfl_tensor = torch.empty((0, 4))
            pred_dfl_tensor = torch.empty((0, 4, self.reg_max))
            pos_mask = torch.zeros(1, dtype=torch.bool)

        targets = {
            "cls_targets": cls_targets,
            "pos_mask": pos_mask,
            "pred_boxes_pos": pred_boxes_tensor,
            "target_boxes_pos": target_boxes_tensor,
            "pred_dfl_pos": pred_dfl_tensor,
            "target_dfl_pos": target_dfl_tensor,
        }

        return img_tensor, targets


def coco_collate_fn(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    b_size = len(batch)
    
    c_p3 = torch.stack([batch[i][1]["cls_targets"][0] for i in range(b_size)], dim=0)
    c_p4 = torch.stack([batch[i][1]["cls_targets"][1] for i in range(b_size)], dim=0)
    c_p5 = torch.stack([batch[i][1]["cls_targets"][2] for i in range(b_size)], dim=0)

    # Concatenate positive detections across all images in the batch
    pos_boxes_target = [b[1]["target_boxes_pos"] for b in batch if b[1]["target_boxes_pos"].shape[0] > 0]
    pos_boxes_pred = [b[1]["pred_boxes_pos"] for b in batch if b[1]["pred_boxes_pos"].shape[0] > 0]
    pos_dfl_target = [b[1]["target_dfl_pos"] for b in batch if b[1]["target_dfl_pos"].shape[0] > 0]
    pos_dfl_pred = [b[1]["pred_dfl_pos"] for b in batch if b[1]["pred_dfl_pos"].shape[0] > 0]

    has_pos = len(pos_boxes_target) > 0
    collated_targets = {
        "cls_targets": [c_p3, c_p4, c_p5],
        "pos_mask": torch.tensor([1 if has_pos else 0], dtype=torch.bool),
        "target_boxes_pos": torch.cat(pos_boxes_target, dim=0) if has_pos else torch.empty((0, 4)),
        "pred_boxes_pos": torch.cat(pos_boxes_pred, dim=0) if has_pos else torch.empty((0, 4)),
        "target_dfl_pos": torch.cat(pos_dfl_target, dim=0) if has_pos else torch.empty((0, 4)),
        "pred_dfl_pos": torch.cat(pos_dfl_pred, dim=0) if has_pos else torch.empty((0, 4, 16)),
    }

    return images, collated_targets