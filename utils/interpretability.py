import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import List, Tuple, Dict, Optional
from models.kan_layers import FastKANLinear
from models.yolo_kan_vlm import YOLOv10_KAN_VLM


def compute_edge_response(
    kan_layer: FastKANLinear,
    in_idx: int,
    out_idx: int,
    x_range: Tuple[float, float] = (-2.0, 2.0),
    num_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluates the continuous 1D learned univariate edge function phi_{j, i}(x):
        phi(x) = w_base * SiLU(x) + sum_k ( c_k * exp(-((x - mu_k) / h)^2) )
    """
    device = kan_layer.grid.device
    x_vals = torch.linspace(x_range[0], x_range[1], num_points, device=device)
    
    with torch.no_grad():
        # Evaluate base activation branch
        base_act = kan_layer.base_act(x_vals)
        # base_linear weight: [out_features, in_features]
        w_b = kan_layer.base_linear.weight[out_idx, in_idx]
        base_out = w_b * base_act

        # Evaluate RBF spline branch
        x_exp = x_vals.unsqueeze(-1)  # [num_points, 1]
        grid = kan_layer.grid.squeeze(0).squeeze(0)  # [grid_size]
        rbf_basis = torch.exp(-((x_exp - grid) * kan_layer.inv_denominator) ** 2)  # [num_points, grid_size]
        
        # spline_weights: [out_features, in_features, grid_size]
        c_k = kan_layer.spline_weights[out_idx, in_idx]  # [grid_size]
        spline_out = torch.matmul(rbf_basis, c_k)

        y_vals = base_out + spline_out

    return x_vals.cpu().numpy(), y_vals.cpu().numpy()


def prune_kan_layer(kan_layer: FastKANLinear, threshold: float = 1e-3) -> int:
    """
    Sparsifies the KAN layer by setting near-zero edge weights to absolute zero.
    Returns the count of pruned (zeroed) univariate edges.
    """
    with torch.no_grad():
        edge_magnitudes = (
            torch.mean(torch.abs(kan_layer.spline_weights), dim=-1)
            + torch.abs(kan_layer.base_linear.weight)
        )
        mask = edge_magnitudes < threshold

        # Zero out both base and spline parameters for inactive connections
        kan_layer.base_linear.weight[mask] = 0.0
        kan_layer.spline_weights[mask] = 0.0

        pruned_count = mask.sum().item()
    return pruned_count


def fit_symbolic_polynomial(
    x_data: np.ndarray,
    y_data: np.ndarray,
    degree: int = 3,
    r2_threshold: float = 0.90,
) -> Tuple[str, float]:
    """
    Fits an analytic polynomial equation to the discrete 1D learned spline response
    using least-squares regression, converting empirical weights into symbolic formulas.
    """
    coeffs = np.polyfit(x_data, y_data, deg=degree)
    poly_func = np.poly1d(coeffs)
    
    # Calculate R^2 coefficient of determination
    y_pred = poly_func(x_data)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    # Convert to clean SymPy mathematical expression
    x = sp.Symbol("x")
    expr = 0
    for power, c in enumerate(coeffs[::-1]):
        if abs(c) > 1e-4:
            expr += round(float(c), 4) * (x ** power)

    symbolic_str = str(sp.simplify(expr))
    return symbolic_str, float(r2)


def plot_top_active_splines(
    kan_layer: FastKANLinear,
    top_k: int = 6,
    save_path: str = "./spline_activations.png",
):
    """
    Identifies and plots the top-K most influential 1D univariate edge activations
    in a KAN layer ranked by their parameter energy.
    """
    with torch.no_grad():
        edge_magnitudes = (
            torch.mean(torch.abs(kan_layer.spline_weights), dim=-1)
            + torch.abs(kan_layer.base_linear.weight)
        )
        flat_indices = torch.topk(edge_magnitudes.view(-1), k=top_k).indices

    rows = (top_k + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = axes.flatten() if top_k > 1 else [axes]

    for idx, flat_idx in enumerate(flat_indices):
        out_i = (flat_idx // kan_layer.in_features).item()
        in_j = (flat_idx % kan_layer.in_features).item()

        x_vals, y_vals = compute_edge_response(kan_layer, in_idx=in_j, out_idx=out_i)
        formula, r2 = fit_symbolic_polynomial(x_vals, y_vals, degree=3)

        ax = axes[idx]
        ax.plot(x_vals, y_vals, label=r"$\phi_{learned}(x)$", color="#1f77b4", lw=2.5)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Edge: In[{in_j}] -> Out[{out_i}]\n$f(x) \\approx {formula}$\n($R^2={r2:.3f}$)", fontsize=10)
        ax.set_xlabel("Input Feature ($x$)")
        ax.set_ylabel(r"Output Response $\phi(x)$")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    # Hide any unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved spline activation plots to: {save_path}")


def explain_class_prediction(
    model: YOLOv10_KAN_VLM,
    image_tensor: torch.Tensor,
    target_class: str,
    all_classes: List[str],
    scale_level: int = 0,
    top_k_channels: int = 5,
) -> Dict[str, any]:
    """
    Computes visual feature contributions for a specific class prediction by tracing
    input channel responses through the KAN multimodal projection layer.
    """
    model.eval()
    device = image_tensor.device
    
    with torch.no_grad():
        text_embeds = model.encode_prompts(all_classes, device)
        features = model.vision_trunk(image_tensor)
        
        # Target scale level (0: P3, 1: P4, 2: P5)
        scale_feat = features[scale_level]
        cls_feat = model.head.cls_convs[scale_level](scale_feat)
        B, C, H, W = cls_feat.shape
        cls_flat = cls_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        kan_layer = model.head.kan_projections[scale_level]
        
        # Mean visual input token vector across spatial locations
        mean_token = cls_flat.mean(dim=1, keepdim=True)  # [1, 1, C]
        
        # Get target class text vector
        class_idx = all_classes.index(target_class)
        target_text_vec = text_embeds[class_idx]  # [Embed_Dim]
        
        # Compute channel-wise contribution to the target class projection
        w_eff = kan_layer.base_linear.weight + torch.mean(kan_layer.spline_weights, dim=-1)
        # Project each input visual channel independently against target text embedding
        # Shape: [C_in]
        channel_contributions = torch.matmul(w_eff.t(), target_text_vec).cpu().numpy()
        
        top_channels = np.argsort(np.abs(channel_contributions))[::-1][:top_k_channels]

    explanations = []
    for c_idx in top_channels:
        x_pts, y_pts = compute_edge_response(kan_layer, in_idx=int(c_idx), out_idx=0)
        formula, r2 = fit_symbolic_polynomial(x_pts, y_pts, degree=3)
        explanations.append({
            "channel_index": int(c_idx),
            "contribution_score": float(channel_contributions[c_idx]),
            "symbolic_formula": formula,
            "fit_r2": r2,
        })

    return {
        "target_class": target_class,
        "scale_level": f"P{scale_level + 3}",
        "top_contributing_channels": explanations,
    }


if __name__ == "__main__":
    # Unit test and execution check
    test_kan = FastKANLinear(in_features=256, out_features=512, grid_size=8)
    
    print("--- 1. Testing KAN Edge Function Extraction ---")
    x, y = compute_edge_response(test_kan, in_idx=0, out_idx=0)
    print(f"Extracted response array shapes: X={x.shape}, Y={y.shape}")

    print("\n--- 2. Testing Symbolic Polynomial Fitting ---")
    eq_str, score = fit_symbolic_polynomial(x, y, degree=3)
    print(f"Learned Equation: f(x) = {eq_str} (R^2: {score:.4f})")

    print("\n--- 3. Testing Edge Pruning ---")
    pruned = prune_kan_layer(test_kan, threshold=0.02)
    print(f"Pruned {pruned} weak spline/linear connections.")

    print("\n--- 4. Generating Visualization Plot ---")
    plot_top_active_splines(test_kan, top_k=6, save_path="./test_splines.png")