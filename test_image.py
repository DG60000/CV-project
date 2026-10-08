import os
import sys
import torch
import numpy as np
from PIL import Image

from models.yolo_detector import YOLOv10Detector
from models.kan_surrogate import KANSurrogate
from models.blip_captioner import BLIPSceneCaptioner
from infer_trustworthy import render_trustworthy_detections


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 75)
    print("  YOLOv10 + KAN Surrogate + BLIP Multimodal Trustworthy AI")
    print("  Research Implementation (arXiv:2603.23037)")
    print("=" * 75)
    print(f"Device: {device}")
    print("Loading models into memory (one-time initialization)...")

    # Load YOLOv10 Detector
    detector = YOLOv10Detector(model_path="yolov10n.pt", conf_threshold=0.25, device=device)

    # Load KAN Surrogate
    kan_surrogate = KANSurrogate(in_features=7, hidden_dim=16, grid_size=10).to(device)
    kan_weights = "checkpoints/kan_surrogate.pth"
    if os.path.exists(kan_weights):
        kan_surrogate.load_state_dict(torch.load(kan_weights, map_location=device))
        print(f"[KAN] Loaded trained surrogate weights from '{kan_weights}'.")
    else:
        print("[KAN] Running with initialized surrogate weights.")
    kan_surrogate.eval()

    # Optional BLIP Captioner
    captioner = None
    use_blip = input("\nEnable BLIP vision-language scene captioner? (y/N): ").strip().lower()
    if use_blip in ["y", "yes"]:
        try:
            captioner = BLIPSceneCaptioner(device=device)
        except Exception as e:
            print(f"[BLIP] Notice: Could not load BLIP ({e}). Proceeding without BLIP.")

    sample_dir = "dataset/train/trainImages"
    sample_images = [f for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png"))] if os.path.exists(sample_dir) else []

    print("\n" + "-" * 75)
    print("Ready! You can now paste image paths.")
    print("Type 'q' or 'exit' anytime to quit.")
    print("-" * 75)

    while True:
        prompt_text = "\nPaste image path (or press Enter for a sample image): "
        raw_path = input(prompt_text).strip()

        if raw_path.lower() in ["q", "exit", "quit"]:
            print("Exiting. Goodbye!")
            break

        # Remove surrounding quotes from Windows drag-and-drop / Copy as path
        clean_path = raw_path.strip('"').strip("'").strip()

        # Fallback to sample image if user simply hits Enter
        if not clean_path:
            if sample_images:
                clean_path = os.path.join(sample_dir, sample_images[0])
                print(f"[Sample] Using default sample: {clean_path}")
            else:
                print("No sample images available. Please paste a valid image path.")
                continue

        if not os.path.exists(clean_path):
            print(f"Error: File does not exist at '{clean_path}'. Please check the path.")
            continue

        try:
            pil_img = Image.open(clean_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image: {e}")
            continue

        print(f"\nProcessing '{clean_path}' (Resolution: {pil_img.size[0]}x{pil_img.size[1]})...")

        # 1. YOLOv10 Detection
        detections = detector.detect(pil_img)
        print(f"[YOLOv10] Detected {len(detections)} candidate objects.")

        # 2. KAN Trustworthiness Evaluation
        if detections:
            feats = np.stack([d["features_7"] for d in detections], axis=0)
            feat_tensor = torch.from_numpy(feats).to(device)
            with torch.no_grad():
                trust_scores = kan_surrogate(feat_tensor).cpu().numpy().flatten()
            for i, d in enumerate(detections):
                d["trust_score"] = float(trust_scores[i])

        # 3. BLIP Scene Caption
        scene_caption = "Scene captioning disabled."
        if captioner is not None:
            scene_caption = captioner.caption_image(pil_img)

        # 4. Print Perception Audit Report
        print("\n" + "=" * 75)
        print(f"  PERCEPTION AUDIT REPORT: {os.path.basename(clean_path)}")
        print("=" * 75)
        if captioner is not None:
            print(f"BLIP Scene Context: \"{scene_caption}\"")
        print(f"Total Detections: {len(detections)}")
        print("-" * 75)
        print(f"{'Class':<15} | {'YOLO Conf':<10} | {'KAN Trust':<10} | {'Status':<12} | {'Note'}")
        print("-" * 75)

        for det in detections:
            c = det["confidence"]
            t = det.get("trust_score", c)
            status = "VERIFIED" if t >= 0.50 else "LOW TRUST"
            note = "High confidence backed by KAN" if t >= 0.50 else "Discrepancy: possible occlusion or blur"
            print(f"{det['class_name']:<15} | {c:<10.3f} | {t:<10.3f} | {status:<12} | {note}")
        print("=" * 75)

        # 5. Render Output Image & Spline Plots
        output_img_path = "./trustworthy_perception.jpg"
        output_plot_path = "./kan_7features_interpretability.png"

        render_trustworthy_detections(
            pil_img,
            detections,
            scene_caption=scene_caption,
            output_path=output_img_path,
            trust_threshold=0.50,
        )
        kan_surrogate.plot_all_splines(save_path=output_plot_path)

        print(f"\n[Artifacts Created]")
        print(f"  1. Annotated Image : {os.path.abspath(output_img_path)}")
        print(f"  2. Spline Curves   : {os.path.abspath(output_plot_path)}")


if __name__ == "__main__":
    main()
