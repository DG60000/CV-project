import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import List, Tuple, Dict, Optional
from .kan_layers import FastKANLinear


FEATURE_NAMES = [
    "x_center",
    "y_center",
    "box_width",
    "box_height",
    "yolo_confidence",
    "class_index",
    "relative_scale",
]

FEATURE_DESCRIPTIONS = [
    "Normalized Bounding-Box Center X (x)",
    "Normalized Bounding-Box Center Y (y)",
    "Normalized Bounding-Box Width (w)",
    "Normalized Bounding-Box Height (h)",
    "Original YOLOv10 Confidence (c)",
    "Discrete Class Index (norm class)",
    "Relative Image Scale (w*h / 640^2)",
]


class KANSurrogate(nn.Module):
    """
    Post-hoc Kolmogorov-Arnold Network (KAN) surrogate model for auditing
    the trustworthiness of YOLOv10 object detections using 7 geometric and semantic features:
        [x, y, w, h, confidence, class_id, scale]
    
    Implements the additive spline-based formulation from Impraimakis et al. (arXiv:2603.23037):
        Trust(x_1, ..., x_7) = sum_{i=1}^7 phi_i(x_i)
    allowing direct univariate inspection, feature importance ranking, and symbolic regression.
    """
    def __init__(
        self,
        in_features: int = 7,
        hidden_dim: Optional[int] = 16,
        grid_size: int = 10,
        grid_min: float = 0.0,
        grid_max: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.grid_size = grid_size
        self.feature_names = FEATURE_NAMES

        if hidden_dim is not None and hidden_dim > 1:
            # 2-layer KAN surrogate for high expressive capacity
            self.kan1 = FastKANLinear(
                in_features=in_features,
                out_features=hidden_dim,
                grid_size=grid_size,
                grid_min=grid_min,
                grid_max=grid_max,
            )
            self.kan2 = FastKANLinear(
                in_features=hidden_dim,
                out_features=1,
                grid_size=grid_size,
                grid_min=grid_min,
                grid_max=grid_max,
            )
            self.is_deep = True
        else:
            # Pure additive 1-layer KAN: output = sum_{i=1}^7 phi_i(x_i)
            self.kan = FastKANLinear(
                in_features=in_features,
                out_features=1,
                grid_size=grid_size,
                grid_min=grid_min,
                grid_max=grid_max,
            )
            self.is_deep = False

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [Batch, 7] tensor of extracted detection features:
                      [x, y, w, h, conf, class_id, scale]
        Returns:
            trust_score: [Batch, 1] bounded trustworthiness prediction in [0, 1]
        """
        if self.is_deep:
            h = self.kan1(features)
            out = self.kan2(h)
        else:
            out = self.kan(features)

        # Trust score is bounded between [0, 1]
        trust = torch.sigmoid(out)
        return trust

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Computes the relative influence / energy of each of the 7 features
        based on the parameter magnitudes of the univariate spline edges.
        """
        primary_kan = self.kan1 if self.is_deep else self.kan
        with torch.no_grad():
            # Sum weight energy per input feature
            spline_energy = torch.mean(torch.abs(primary_kan.spline_weights), dim=-1)  # [out_features, in_features]
            base_energy = torch.abs(primary_kan.base_linear.weight)
            total_energy = (spline_energy + base_energy).sum(dim=0).cpu().numpy()  # [in_features]

            total_sum = total_energy.sum() + 1e-8
            normalized_pct = (total_energy / total_sum) * 100.0

        return {
            self.feature_names[i]: float(normalized_pct[i])
            for i in range(self.in_features)
        }

    def evaluate_feature_spline(
        self,
        feature_idx: int,
        x_min: float = 0.0,
        x_max: float = 1.0,
        num_points: int = 200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates the 1D continuous univariate edge function phi_i(x) for input feature i.
        """
        primary_kan = self.kan1 if self.is_deep else self.kan
        device = primary_kan.grid.device
        x_vals = torch.linspace(x_min, x_max, num_points, device=device)

        with torch.no_grad():
            # Base activation contribution
            base_act = primary_kan.base_act(x_vals)
            # Average base weight across output features for this input
            w_b = primary_kan.base_linear.weight[:, feature_idx].mean()
            base_out = w_b * base_act

            # RBF spline basis expansion
            x_exp = x_vals.unsqueeze(-1)
            grid = primary_kan.grid.squeeze(0).squeeze(0)
            rbf_basis = torch.exp(-((x_exp - grid) * primary_kan.inv_denominator) ** 2)

            # Average spline coefficients for this input
            c_k = primary_kan.spline_weights[:, feature_idx, :].mean(dim=0)
            spline_out = torch.matmul(rbf_basis, c_k)

            y_vals = base_out + spline_out

        return x_vals.cpu().numpy(), y_vals.cpu().numpy()

    def fit_symbolic_polynomial(
        self,
        feature_idx: int,
        degree: int = 3,
    ) -> Tuple[str, float]:
        """
        Fits a symbolic polynomial formula f_i(x) to the learned spline curve
        using least-squares regression.
        """
        x_vals, y_vals = self.evaluate_feature_spline(feature_idx)
        coeffs = np.polyfit(x_vals, y_vals, deg=degree)
        poly_func = np.poly1d(coeffs)

        y_pred = poly_func(x_vals)
        ss_res = np.sum((y_vals - y_pred) ** 2)
        ss_tot = np.sum((y_vals - np.mean(y_vals)) ** 2)
        r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

        x = sp.Symbol("x")
        expr = 0
        for power, c in enumerate(coeffs[::-1]):
            if abs(c) > 1e-4:
                expr += round(float(c), 4) * (x ** power)

        symbolic_str = str(sp.simplify(expr))
        return symbolic_str, float(r2)

    def plot_all_splines(
        self,
        save_path: str = "./kan_7features_interpretability.png",
    ):
        """
        Generates a 7-panel visualization displaying the exact learned univariate
        spline curve phi_i(x_i) and symbolic polynomial for all 7 features, matching
        the interpretability figures in the research paper.
        """
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes = axes.flatten()

        importances = self.get_feature_importance()

        for i in range(self.in_features):
            ax = axes[i]
            feat_name = self.feature_names[i]
            feat_desc = FEATURE_DESCRIPTIONS[i]
            pct = importances[feat_name]

            x_pts, y_pts = self.evaluate_feature_spline(i)
            formula, r2 = self.fit_symbolic_polynomial(i, degree=3)

            ax.plot(x_pts, y_pts, color="#1f77b4", lw=2.5, label=r"$\phi_{learned}(x)$")
            ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
            ax.set_title(
                f"Feature {i+1}: {feat_name} ({pct:.1f}% Importance)\n"
                f"${formula}$\n($R^2={r2:.3f}$)",
                fontsize=9,
            )
            ax.set_xlabel(f"{feat_desc}")
            ax.set_ylabel(r"Univariate Response $\phi(x)$")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)

        # Plot overall feature importance ranking in panel 8
        ax_imp = axes[7]
        names = [self.feature_names[i] for i in range(self.in_features)]
        vals = [importances[name] for name in names]
        colors = ["#2ca02c" if "conf" in n or "class" in n else "#1f77b4" for n in names]
        bars = ax_imp.barh(names, vals, color=colors)
        ax_imp.set_title("Overall Feature Importance Ranking (%)", fontsize=10)
        ax_imp.set_xlabel("Contribution (%)")
        ax_imp.grid(True, alpha=0.3)
        for bar, val in zip(bars, vals):
            ax_imp.text(val + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=8)

        # Hide 9th panel
        fig.delaxes(axes[8])

        plt.suptitle(
            "Kolmogorov-Arnold Network (KAN) 7-Feature Interpretability Audit\n"
            "arXiv:2603.23037 - Impraimakis et al.",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[KAN Interpretability] Saved 7-feature spline explanation plot to: {save_path}")


def extract_7_features(
    box_xywh_norm: torch.Tensor,
    confidence: torch.Tensor,
    class_id: torch.Tensor,
    num_classes: int = 80,
    img_size: int = 640,
) -> torch.Tensor:
    """
    Extracts the exact 7 geometric and semantic features defined in the paper:
      1. x: normalized bounding-box center X
      2. y: normalized bounding-box center Y
      3. w: normalized bounding-box width
      4. h: normalized bounding-box height
      5. c: predicted confidence
      6. class_index: normalized class index (class_id / num_classes)
      7. relative_scale: (w * h * img_size^2) / img_size^2 = w * h
    
    Args:
        box_xywh_norm: Tensor of shape [N, 4] with normalized (cx, cy, w, h) in [0, 1]
        confidence: Tensor of shape [N] or [N, 1] with YOLO confidence scores
        class_id: Tensor of shape [N] or [N, 1] with class indices
        num_classes: Total class count (e.g., 80 for COCO)
        img_size: Input resolution (default 640)
    Returns:
        features: Tensor of shape [N, 7]
    """
    if confidence.dim() == 1:
        confidence = confidence.unsqueeze(-1)
    if class_id.dim() == 1:
        class_id = class_id.unsqueeze(-1)

    x = box_xywh_norm[:, 0:1]
    y = box_xywh_norm[:, 1:2]
    w = box_xywh_norm[:, 2:3]
    h = box_xywh_norm[:, 3:4]
    c = confidence.float()
    cls_norm = (class_id.float() / float(num_classes)).clamp(0.0, 1.0)
    
    # Relative scale: (w_pixels * h_pixels) / 640^2 = w * h
    scale = (w * h).clamp(0.0, 1.0)

    features = torch.cat([x, y, w, h, c, cls_norm, scale], dim=-1)
    return features
