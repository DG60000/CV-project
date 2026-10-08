# YOLO with Kolmogorov-Arnold Networks (KAN) & Vision-Language Foundation Models (VLM)
### Implementation of arXiv:2603.23037 (*Scientific Reports*, 2026)
**Authors:** Marios Impraimakis, Daniel Vazquez, Feiyu Zhou

---

## 📌 1. Overview of the Research Paper

Modern object detectors like **YOLOv10** generate bounding boxes and confidence scores, but confidence scores act as "black-box" numbers. In visually degraded scenarios (such as **occlusion**, **motion blur**, **low lighting**, or **texture loss**), YOLO can output high confidence on incorrect or hallucinated detections.

This framework introduces a **two-tier trustworthy multimodal AI perception system**:
1. **Perception Engine (YOLOv10):** Real-time object detection producing candidate bounding boxes.
2. **Post-Hoc KAN Surrogate Audit:** A Kolmogorov-Arnold Network (KAN) trained as an interpretable surrogate on **seven geometric and semantic features** to evaluate prediction trustworthiness.
3. **Multimodal Grounding (BLIP):** A Bootstrapped Language-Image Pre-training (BLIP) foundation model providing human-understandable natural language scene descriptions.
4. **Symbolic Spline Interpretability:** Univariate continuous spline functions $\phi_i(x_i)$ enabling direct visualization and SymPy mathematical equation fitting for every feature.

---

## 📐 2. The Seven Features Extracted for KAN

For every candidate detection output by YOLOv10, the pipeline extracts exactly seven normalized features:

| # | Feature | Definition | Description |
|---|---|---|---|
| **1** | $x$ | $x_{\text{center}} / W_{\text{orig}}$ | Normalized bounding-box center X coordinate $\in [0, 1]$ |
| **2** | $y$ | $y_{\text{center}} / H_{\text{orig}}$ | Normalized bounding-box center Y coordinate $\in [0, 1]$ |
| **3** | $w$ | $\text{width} / W_{\text{orig}}$ | Normalized bounding-box width $\in [0, 1]$ |
| **4** | $h$ | $\text{height} / H_{\text{orig}}$ | Normalized bounding-box height $\in [0, 1]$ |
| **5** | $c$ | YOLO Confidence | Raw YOLOv10 output confidence score $\in [0, 1]$ |
| **6** | $\text{class\_id}$ | $\text{id} / N_{\text{classes}}$ | Normalized discrete object class category |
| **7** | $s$ | $(w_{\text{px}} \cdot h_{\text{px}}) / 640^2 = w \cdot h$ | Relative image bounding box scale |

### Key Empirical Finding from the Paper:
> **Class index and confidence are the primary drivers of trustworthiness**, while spatial coordinates ($x, y, w, h$) exert secondary, moderating effects.

---

## 🚀 3. Quick Start & Execution

### A. Run Trustworthy Inference & Audit
Run perception on any test image to generate detection bounding boxes, trust audit scores, and the 7-spline explanation plot:
```bash
python infer_trustworthy.py --image dataset/train/trainImages/000000119233.jpg
```
*(Tip: Add `--no_blip` for ultra-fast offline execution without the BLIP captioning model).*

#### Output Artifacts:
1. **Annotated Image (`trustworthy_perception.jpg`):**
   - Displays bounding boxes with both raw YOLO confidence and KAN Trust Score.
   - Color-coded: **Green (`[TRUSTED]`)** vs. **Red (`[LOW-TRUST / AMBIGUOUS]`)**.
   - Top banner with BLIP natural language scene context.
2. **7-Spline Explanation Figure (`kan_7features_interpretability.png`):**
   - 7 panels showing continuous learned spline functions $\phi_i(x_i)$.
   - SymPy symbolic polynomial formula $f(x)$ and $R^2$ fit for each feature.
   - Summary bar chart showing relative percentage contribution of each feature.

---

### B. Train / Fine-Tune the KAN Surrogate
To re-train the KAN surrogate model on your COCO dataset:
```bash
python train_surrogate.py --max_images 200 --epochs 30
```
- Automatically extracts 7-features from YOLOv10 detections.
- Caches features to `checkpoints/features_7.npz` for instantaneous subsequent re-runs.
- Evaluates $R^2$ fidelity score across training epochs ($R^2 > 0.98$).
- Exports weights to `checkpoints/kan_surrogate.pth`.

---

## 🏗️ 4. Repository Structure

```
yolov10_wan_vln/
├── models/
│   ├── kan_surrogate.py      # 7-feature KAN surrogate model & symbolic fitting
│   ├── yolo_detector.py      # YOLOv10 detection wrapper with 7-feature extraction
│   ├── blip_captioner.py     # BLIP vision-language foundation model module
│   ├── kan_layers.py         # FastKANLinear with Gaussian RBF bases
│   ├── heads.py              # Multimodal detection heads
│   └── yolo_kan_vlm.py       # End-to-end YOLO-KAN-VLM model architecture
├── utils/
│   ├── dataset.py            # COCO JSON and TXT annotation loaders
│   ├── interpretability.py   # Spline curve extraction, pruning, SymPy regression
│   └── losses.py             # CIoU, DFL, and KAN spline L1 sparsity losses
├── configs/
│   └── model_config.yaml     # System hyperparameter configuration
├── checkpoints/
│   ├── kan_surrogate.pth     # Trained KAN surrogate model weights
│   └── features_7.npz        # Pre-extracted 7-feature dataset cache
├── infer_trustworthy.py      # Complete end-to-end inference and audit pipeline
├── train_surrogate.py        # Post-hoc KAN surrogate training script
├── train.py                  # End-to-end model training script
└── requirements.txt          # Python dependencies
```
