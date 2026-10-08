import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from models.kan_surrogate import KANSurrogate, extract_7_features, FEATURE_NAMES
from models.yolo_detector import YOLOv10Detector


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Computes Intersection-over-Union (IoU) between two xyxy bounding boxes.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def load_coco_annotations(ann_file: str) -> Dict[str, List[Dict[str, any]]]:
    """
    Loads COCO instances json and indexes ground-truth boxes by image file name.
    """
    print(f"[Dataset] Parsing COCO annotations from {ann_file}...")
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    # Map image_id -> file_name
    id_to_filename = {}
    id_to_dims = {}
    for img in coco_data["images"]:
        id_to_filename[img["id"]] = img["file_name"]
        id_to_dims[img["id"]] = (img["width"], img["height"])

    # Map file_name -> list of annotations [x1, y1, x2, y2, category_id]
    annotations_by_file = {}
    for ann in coco_data["annotations"]:
        img_id = ann["image_id"]
        fname = id_to_filename.get(img_id)
        if fname is None:
            continue

        if fname not in annotations_by_file:
            annotations_by_file[fname] = []

        # COCO bbox: [x, y, width, height] in pixel coordinates
        bx, by, bw, bh = ann["bbox"]
        xyxy = [bx, by, bx + bw, by + bh]
        cat_id = ann["category_id"]

        annotations_by_file[fname].append({
            "bbox_xyxy": xyxy,
            "category_id": cat_id,
        })

    print(f"[Dataset] Indexed annotations for {len(annotations_by_file)} images.")
    return annotations_by_file


def collect_surrogate_dataset(
    detector: YOLOv10Detector,
    images_dir: str,
    annotations: Optional[Dict[str, List[Dict[str, any]]]] = None,
    max_images: int = 500,
    iou_match_thresh: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs YOLOv10 over images, extracts the 7 features for each detected box,
    and assigns ground-truth trustworthiness labels based on IoU with COCO targets.
    """
    img_files = [
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    img_files.sort()
    sample_files = img_files[:max_images]

    print(f"[Feature Extraction] Processing {len(sample_files)} images through YOLOv10...")
    all_features = []
    all_targets = []

    for fname in tqdm(sample_files, desc="Extracting 7-Features"):
        img_path = os.path.join(images_dir, fname)
        try:
            pil_img = Image.open(img_path).convert("RGB")
        except Exception as e:
            continue

        detections = detector.detect(pil_img, conf=0.15)
        gt_list = annotations.get(fname, []) if annotations is not None else []

        for det in detections:
            feat_7 = det["features_7"]  # [x, y, w, h, conf, class_id, scale]
            det_box = det["box_xyxy"]
            conf = det["confidence"]

            # Ground truth trustworthiness assignment
            if gt_list:
                max_iou = 0.0
                for gt in gt_list:
                    iou = compute_iou(det_box, gt["bbox_xyxy"])
                    if iou > max_iou:
                        max_iou = iou

                # Trustworthy if IoU >= 0.5; penalize hallucination / overconfidence
                if max_iou >= iou_match_thresh:
                    # High trust: calibrated confidence
                    target_trust = float(max_iou * conf)
                else:
                    # Low trust: false positive / hallucination / occlusion mismatch
                    target_trust = 0.0
            else:
                # Self-calibration: fit original confidence distribution
                target_trust = conf

            all_features.append(feat_7)
            all_targets.append(target_trust)

    X = np.array(all_features, dtype=np.float32)
    Y = np.array(all_targets, dtype=np.float32).reshape(-1, 1)
    print(f"[Feature Extraction] Collected {len(X)} detection samples with 7 features.")
    return X, Y


def train_kan_surrogate(
    X: np.ndarray,
    Y: np.ndarray,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    l1_weight: float = 1e-4,
    device: str = "cpu",
) -> KANSurrogate:
    """
    Trains the KAN surrogate model to model trustworthiness from the 7 features.
    """
    model = KANSurrogate(in_features=7, hidden_dim=16, grid_size=10).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("\n--- Training KAN Surrogate Model (Impraimakis et al., 2026) ---")
    model.train()
    for ep in range(1, epochs + 1):
        running_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()

            pred = model(bx)
            loss_mse = criterion(pred, by)

            # L1 penalty on spline weights for interpretability-driven sparsity
            l1_penalty = model.kan1.get_spline_l1_reg()
            if hasattr(model, "kan2"):
                l1_penalty += model.kan2.get_spline_l1_reg()

            loss = loss_mse + l1_weight * l1_penalty
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        if ep % 5 == 0 or ep == epochs:
            # Evaluate R^2 on dataset
            model.eval()
            with torch.no_grad():
                preds_all = model(torch.from_numpy(X).to(device)).cpu().numpy()
            ss_res = np.sum((Y - preds_all) ** 2)
            ss_tot = np.sum((Y - np.mean(Y)) ** 2)
            r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
            print(f"Epoch [{ep:2d}/{epochs:2d}] | MSE Loss: {running_loss/len(loader):.5f} | R² Fidelity: {r2:.4f}")
            model.train()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train KAN Post-Hoc Surrogate for YOLOv10")
    parser.add_argument("--images_dir", type=str, default="dataset/train/trainImages", help="Directory of images")
    parser.add_argument("--ann_file", type=str, default="dataset/train/annotations/instances_train2017.json", help="COCO annotations")
    parser.add_argument("--max_images", type=int, default=200, help="Number of images to sample")
    parser.add_argument("--cache_file", type=str, default="checkpoints/features_7.npz", help="Cache npz path")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save_path", type=str, default="checkpoints/kan_surrogate.pth", help="Checkpoint save path")
    parser.add_argument("--plot_path", type=str, default="kan_7features_interpretability.png", help="Plot save path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Feature Collection or Load Cache
    if os.path.exists(args.cache_file):
        print(f"[Cache] Loading pre-extracted features from {args.cache_file}...")
        data = np.load(args.cache_file)
        X, Y = data["X"], data["Y"]
    else:
        detector = YOLOv10Detector("yolov10n.pt", device=device)
        annotations = None
        if os.path.exists(args.ann_file):
            annotations = load_coco_annotations(args.ann_file)
        X, Y = collect_surrogate_dataset(detector, args.images_dir, annotations, max_images=args.max_images)
        os.makedirs(os.path.dirname(args.cache_file), exist_ok=True)
        np.savez_compressed(args.cache_file, X=X, Y=Y)
        print(f"[Cache] Saved extracted features to {args.cache_file}")

    # 2. Train KAN Surrogate
    model = train_kan_surrogate(X, Y, epochs=args.epochs, lr=args.lr, device=device)

    # 3. Save Checkpoint
    torch.save(model.state_dict(), args.save_path)
    print(f"\n[Checkpoint] Saved trained KAN surrogate weights to {args.save_path}")

    # 4. Feature Importance & Symbolic Polynomials
    print("\n--- KAN Surrogate Feature Importance Breakdown ---")
    importances = model.get_feature_importance()
    for feat_name, pct in importances.items():
        print(f"  - {feat_name:18s}: {pct:5.2f}%")

    # 5. Plot 7-Spline Interpretability
    model.plot_all_splines(save_path=args.plot_path)


if __name__ == "__main__":
    main()
