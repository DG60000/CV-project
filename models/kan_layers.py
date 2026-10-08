import torch
import torch.nn as nn
import torch.nn.functional as F


class FastKANLinear(nn.Module):
    """
    Fast Kolmogorov-Arnold Network Layer using Radial Basis Functions (RBF).
    Replaces static node activations with learnable univariate edge functions:
        phi(x) = w_base * act(x) + sum(c_k * RBF_k(x))
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        base_activation=nn.SiLU,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        # Residual standard linear transformation
        self.base_linear = nn.Linear(in_features, out_features, bias=False)
        self.base_act = base_activation()

        # Fixed Gaussian RBF centers over the grid interval
        grid = torch.linspace(grid_min, grid_max, grid_size).unsqueeze(0).unsqueeze(0)
        self.register_buffer("grid", grid)  # Shape: [1, 1, grid_size]
        self.inv_denominator = 1.0 / (grid[0, 0, 1] - grid[0, 0, 0])

        # Learnable RBF spline transformation coefficients
        self.spline_weights = nn.Parameter(
            torch.empty(out_features, in_features, grid_size).normal_(0.0, 0.05)
        )
        self.base_weights = nn.Parameter(torch.ones(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [Batch, In_Features] or [Batch, Tokens, In_Features]
        Returns:
            Projected output tensor of shape [Batch, Out_Features] or [Batch, Tokens, Out_Features]
        """
        is_2d = (x.dim() == 2)
        if is_2d:
            x = x.unsqueeze(1)  # [B, 1, In_Features]

        # Base residual pathway: [B, N, C_in] -> [B, N, C_out]
        base_output = self.base_linear(self.base_act(x))

        # RBF basis expansion: exp(-((x - mu) / h)^2)
        x_expanded = x.unsqueeze(-1)  # [B, N, C_in, 1]
        rbf_basis = torch.exp(-((x_expanded - self.grid) * self.inv_denominator) ** 2)  # [B, N, C_in, Grid]

        # Spline contraction over input features and RBF grid nodes
        spline_output = torch.einsum("bnig,oig->bno", rbf_basis, self.spline_weights)

        out = base_output + spline_output
        if is_2d:
            return out.squeeze(1)
        return out

    def get_spline_l1_reg(self) -> torch.Tensor:
        """Computes L1 norm over spline weights for interpretability-driven sparsity."""
        return torch.mean(torch.abs(self.spline_weights))