import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict

from models.yolo_kan_vlm import YOLOv10_KAN_VLM
from utils.interpretability import explain_class_prediction, plot_top_active_splines


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """
    Resizes and pads image to match target input dimensions while preserving aspect ratio.
    """
    shape = img.shape[:2]  # [height, width]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, r, (dw, dh)


def decode_boxes(
    reg_preds: List[torch.Tensor],
    strides: List[int] = [8, 16, 32],
    reg_max: int = 16,
) -> torch.Tensor:
    """
    Decodes Distribution Focal Loss (DFL) outputs to (x1, y1, x2, y2) bounding box coordinates.
    """
    project = torch.linspace(0, reg_max - 1, reg_max, device=reg_preds[0].device)
    all_boxes = []

    for reg, stride in zip(reg_preds, strides):
        B, _, H, W = reg.shape
        # Reshape to [B, H, W, 4, reg_max]
        reg = reg.view(B, 4, reg_max, H, W).permute(0, 3, 4, 1, 2)
        # Softmax expectation over distribution bins
        dist = F.softmax(reg, dim=-1).matmul(project)  # [B, H, W, 4] (left, top, right, bottom)
        dist = dist * stride

        # Create coordinate anchor grid
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=reg.device),
            torch.arange(W, device=reg.device),
            indexing="ij"
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).float() * stride  # [H, W, 2]
        grid = grid.unsqueeze(0)  # [1, H, W, 2]

        x1 = grid[..., 0] - dist[..., 0]
        y1 = grid[..., 1] - dist[..., 1]
        x2 = grid[..., 0] + dist[..., 2]
        y2 = grid[..., 1] + dist[..., 3]

        boxes = torch.stack([x1, y1, x2, y2], dim=-1).view(B, -1, 4)
        all_boxes.append(boxes)

    return torch.cat(all_boxes, dim=1)  # [B, Total_Anchors, 4]


def postprocess(
    cls_logits: List[torch.Tensor],
    reg_preds: List[torch.Tensor],
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.45,
    strides: List[int] = [8, 16, 32],
) -> List[Dict[str, torch.Tensor]]:
    """
    Decodes predictions, computes class confidences, and applies Non-Maximum Suppression (NMS).
    """
    decoded_boxes = decode_boxes(reg_preds, strides=strides)  # [B, N, 4]

    # Flatten and concatenate multi-scale classification logits
    all_cls = []
    for cls_map in cls_logits:
        B, C, H, W = cls_map.shape
        cls_flat = cls_map.permute(0, 2, 3, 1).reshape(B, H * W, C)
        all_cls.append(cls_flat)
    
    cls_scores = torch.sigmoid(torch.cat(all_cls, dim=1))  # [B, N, Num_Classes]

    batch_detections = []
    for b in range(decoded_boxes.shape[0]):
        boxes = decoded_boxes[b]
        scores = cls_scores[b]

        max_scores, class_ids = scores.max(dim=-1)
        mask = max_scores > conf_thresh

        valid_boxes = boxes[mask]
        valid_scores = max_scores[mask]
        valid_classes = class_ids[mask]

        if valid_boxes.numel() == 0:
            batch_detections.append({
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty((0,)),
                "classes": torch.empty((0,), dtype=torch.long),
            })
            continue

        # Non-Maximum Suppression (NMS)
        keep = torch.ops.torchvision.nms(valid_boxes, valid_scores, iou_thresh)
        batch_detections.append({
            "boxes": valid_boxes[keep],
            "scores": valid_scores[keep],
            "classes": valid_classes[keep],
        })

    return batch_detections


def draw_detections(
    image_path: str,
    detections: Dict[str, torch.Tensor],
    class_names: List[str],
    ratio: float,
    padding: Tuple[float, float],
    output_path: str,
):
    """
    Renders bounding boxes and predicted text prompts onto the source image.
    """
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    dw, dh = padding

    boxes = detections["boxes"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()
    classes = detections["classes"].cpu().numpy()

    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3)).tolist()

    for box, score, cls_id in zip(boxes, scores, classes):
        # Rescale coordinates to original image size
        x1 = (box[0] - dw) / ratio
        y1 = (box[1] - dh) / ratio
        x2 = (box[2] - dw) / ratio
        y2 = (box[3] - dh) / ratio

        label = f"{class_names[cls_id]}: {score:.2f}"
        color = tuple(colors[cls_id % len(colors)])

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 4, y1 + 4), label, fill=color)

    image.save(output_path)
    print(f"Detections successfully saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Inference for YOLOv10-KAN-VLM")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--classes", type=str, nargs="+", default=["person", "car", "dog"], help="Target prompt classes")
    parser.add_argument("--weights", type=str, default=None, help="Path to saved model checkpoint (.pth)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    parser.add_argument("--output", type=str, default="./output.jpg", help="Path to save annotated image")
    parser.add_argument("--explain", action="store_true", help="Generate KAN interpretability reports for detected classes")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Model Architecture
    model = YOLOv10_KAN_VLM(
        grid_size=8,
        embed_dim=512,
        clip_model_name="openai/clip-vit-base-patch32",
    ).to(device)

    if args.weights and os.path.exists(args.weights):
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded weights from {args.weights}")
    else:
        print("Running with initialized model weights (unfine-tuned / zero-shot test).")

    model.eval()

    # 2. Image Preprocessing
    orig_img = cv2.imread(args.image)
    if orig_img is None:
        raise FileNotFoundError(f"Could not load image from {args.image}")
        
    padded_img, ratio, padding = letterbox(orig_img, new_shape=(args.imgsz, args.imgsz))
    input_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    input_tensor = input_tensor.to(device)

    # 3. Model Inference
    with torch.no_grad():
        cls_logits, reg_preds = model(input_tensor, args.classes)
        detections = postprocess(cls_logits, reg_preds, conf_thresh=args.conf, iou_thresh=args.iou)[0]

    # 4. Save Rendered Output
    draw_detections(args.image, detections, args.classes, ratio, padding, args.output)

    # 5. Optional KAN Explainability Analysis
    if args.explain and len(detections["classes"]) > 0:
        print("\n--- Generating KAN Interpretability Analysis ---")
        detected_class_names = list(set([args.classes[c] for c in detections["classes"].cpu().numpy()]))
        
        for cls_name in detected_class_names:
            explanation = explain_class_prediction(
                model=model,
                image_tensor=input_tensor,
                target_class=cls_name,
                all_classes=args.classes,
                scale_level=0,
                top_k_channels=3,
            )
            print(f"\nTarget Class: '{cls_name}' on Feature Scale {explanation['scale_level']}:")
            for ch in explanation["top_contributing_channels"]:
                print(f"  - Channel [{ch['channel_index']}]: Score = {ch['contribution_score']:.4f} | Formula: f(x) ≈ {ch['symbolic_formula']} (R²={ch['fit_r2']:.3f})")

        # Plot active spline curves of the primary detection head
        plot_top_active_splines(model.head.kan_projections[0], top_k=6, save_path="./kan_spline_explanation.png")


if __name__ == "__main__":
    main()