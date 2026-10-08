import os
import argparse
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional

from models.yolo_detector import YOLOv10Detector
from models.kan_surrogate import KANSurrogate, FEATURE_NAMES, FEATURE_DESCRIPTIONS
from models.blip_captioner import BLIPSceneCaptioner


def render_trustworthy_detections(
    image: Image.Image,
    detections: List[Dict[str, any]],
    scene_caption: str,
    output_path: str = "./trustworthy_detection_output.jpg",
    trust_threshold: float = 0.50,
):
    """
    Renders bounding boxes annotated with:
      - Class name
      - Raw YOLOv10 confidence (c)
      - KAN Surrogate Trust Score (T)
      - Trust Status: [VERIFIED] in Green vs [LOW TRUST / AMBIGUOUS] in Red
      - BLIP Scene Caption banner across the top
    """
    img_draw = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_draw)
    width, height = img_draw.size

    # Try loading a readable default font
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, int(height * 0.022)))
        header_font = ImageFont.truetype("arial.ttf", size=max(16, int(height * 0.028)))
    except Exception:
        font = ImageFont.load_default()
        header_font = font

    # Draw top banner for BLIP natural language caption
    banner_height = max(40, int(height * 0.06))
    draw.rectangle([0, 0, width, banner_height], fill=(20, 24, 33))
    caption_text = f"BLIP VLM Scene Context: \"{scene_caption}\""
    draw.text((12, int(banner_height * 0.25)), caption_text, fill=(240, 240, 255), font=header_font)

    for det in detections:
        box = det["box_xyxy"]
        x1, y1, x2, y2 = box
        c = det["confidence"]
        trust = det.get("trust_score", c)
        cls_name = det["class_name"]

        is_trustworthy = (trust >= trust_threshold)
        # Green for verified, Red/Orange for low-trust or overconfident
        box_color = (46, 204, 113) if is_trustworthy else (231, 76, 60)
        status_tag = "TRUSTED" if is_trustworthy else "LOW-TRUST"

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        # Label tag
        label = f"{cls_name} | YOLO: {c:.2f} | KAN Trust: {trust:.2f} [{status_tag}]"
        bbox = draw.textbbox((x1, y1 - 22), label, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=box_color)
        draw.text((x1, y1 - 22), label, fill=(255, 255, 255), font=font)

    img_draw.save(output_path, quality=95)
    print(f"[Inference] Saved annotated perception image to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Trustworthy Object Detection Inference with YOLOv10, KAN Surrogate, and BLIP VLM"
    )
    parser.add_argument("--image", type=str, default=None, help="Path to input test image (if omitted, you will be prompted to paste it)")
    parser.add_argument("--yolo_weights", type=str, default="yolov10n.pt", help="YOLOv10 model weights")
    parser.add_argument("--kan_weights", type=str, default="checkpoints/kan_surrogate.pth", help="KAN surrogate checkpoint")
    parser.add_argument("--conf_thresh", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--trust_thresh", type=float, default=0.50, help="KAN trust audit threshold")
    parser.add_argument("--output_image", type=str, default="./trustworthy_perception.jpg", help="Output annotated image")
    parser.add_argument("--output_plot", type=str, default="./kan_7features_interpretability.png", help="Output spline plot")
    parser.add_argument("--no_blip", action="store_true", help="Skip BLIP captioning for fast offline run")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[System] Running inference on device: {device}")

    # Prompt user to paste path if not passed as CLI argument
    image_path = args.image
    if not image_path:
        print("\n" + "="*60)
        print("  YOLOv10 + KAN + VLM Trustworthy Object Detection")
        print("="*60)
        raw_input = input("Paste the path of your image (or drag & drop here): ").strip()
        # Clean Windows pasted quotes (e.g., "C:\path\to\image.jpg")
        image_path = raw_input.strip('"').strip("'").strip()

    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: '{image_path}'")
    pil_image = Image.open(image_path).convert("RGB")
    print(f"[Image] Loaded '{image_path}' (Resolution: {pil_image.size[0]}x{pil_image.size[1]})")

    # 2. YOLOv10 Perception Stage
    detector = YOLOv10Detector(model_path=args.yolo_weights, conf_threshold=args.conf_thresh, device=device)
    detections = detector.detect(pil_image)
    print(f"[YOLOv10] Extracted {len(detections)} candidate object detections.")

    # 3. KAN Post-Hoc Surrogate Trustworthiness Audit Stage
    kan_surrogate = KANSurrogate(in_features=7, hidden_dim=16, grid_size=10).to(device)
    if os.path.exists(args.kan_weights):
        kan_surrogate.load_state_dict(torch.load(args.kan_weights, map_location=device))
        print(f"[KAN Surrogate] Loaded trained weights from {args.kan_weights}")
    else:
        print("[KAN Surrogate] Warning: No trained weights found at specified path. Running with initialized surrogate.")

    kan_surrogate.eval()

    if detections:
        # Batch 7 features into tensor
        features_batch = np.stack([det["features_7"] for det in detections], axis=0)
        feat_tensor = torch.from_numpy(features_batch).to(device)

        with torch.no_grad():
            trust_scores = kan_surrogate(feat_tensor).cpu().numpy().flatten()

        for i, det in enumerate(detections):
            det["trust_score"] = float(trust_scores[i])

    # 4. BLIP Vision-Language Foundation Model Scene Context
    scene_caption = "Scene description disabled."
    if not args.no_blip:
        try:
            captioner = BLIPSceneCaptioner(device=device)
            scene_caption = captioner.caption_image(pil_image)
            print(f"[BLIP Scene Caption] \"{scene_caption}\"")
        except Exception as e:
            print(f"[BLIP] Notice: Could not initialize BLIP ({e}). Proceeding without caption banner.")
            scene_caption = "Natural language caption unavailable"

    # 5. Generate Multimodal Trustworthiness Report
    print("\n" + "="*80)
    print("  TRUSTWORTHY MULTIMODAL PERCEPTION AUDIT REPORT (arXiv:2603.23037)")
    print("="*80)
    print(f"Scene Linguistic Context (BLIP): \"{scene_caption}\"")
    print(f"Total Detections: {len(detections)}")
    print("-"*80)
    print(f"{'Class':<15} | {'YOLO Conf':<10} | {'KAN Trust':<10} | {'Status':<12} | {'Note'}")
    print("-"*80)

    for det in detections:
        c = det["confidence"]
        t = det.get("trust_score", c)
        status = "VERIFIED" if t >= args.trust_thresh else "LOW TRUST"
        note = "High confidence backed by KAN" if t >= args.trust_thresh else "Discrepancy: possible occlusion or blur"
        print(f"{det['class_name']:<15} | {c:<10.3f} | {t:<10.3f} | {status:<12} | {note}")
    print("="*80 + "\n")

    # 6. Render Output Image
    render_trustworthy_detections(
        pil_image,
        detections,
        scene_caption=scene_caption,
        output_path=args.output_image,
        trust_threshold=args.trust_thresh,
    )

    # 7. Render 7-Spline KAN Interpretability Curves
    kan_surrogate.plot_all_splines(save_path=args.output_plot)


if __name__ == "__main__":
    main()
