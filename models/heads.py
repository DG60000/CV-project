import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from .kan_layers import FastKANLinear


class ConvBNAct(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class KANMultimodalDetectHead(nn.Module):
    """
    Decoupled YOLO Detection Head with:
      1. Standard Conv stacks for bounding box regression & Distribution Focal Loss (DFL).
      2. KAN projection heads aligning spatial region features to VLM semantic text embeddings.
    """
    def __init__(
        self,
        in_channels: List[int] = [256, 512, 1024],
        embed_dim: int = 512,
        reg_max: int = 16,
        grid_size: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.reg_max = reg_max

        # Bounding Box Regression Branches (P3, P4, P5)
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(c, c, 3, 1, 1),
                ConvBNAct(c, c, 3, 1, 1),
                nn.Conv2d(c, 4 * reg_max, kernel_size=1)
            ) for c in in_channels
        ])

        # Feature preparation prior to KAN projection
        self.cls_convs = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(c, c, 3, 1, 1),
                ConvBNAct(c, c, 3, 1, 1)
            ) for c in in_channels
        ])

        # KAN non-linear projections mapping visual feature dimension -> VLM embed_dim
        self.kan_projections = nn.ModuleList([
            FastKANLinear(in_features=c, out_features=embed_dim, grid_size=grid_size)
            for c in in_channels
        ])

        # Learnable temperature parameter for cosine similarity
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1.0 / 0.07).log())

    def forward(
        self, feats: List[torch.Tensor], text_embeddings: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            feats: Multiscale feature list [P3, P4, P5] from backbone/PAN.
            text_embeddings: [Num_Classes, Embed_Dim] normalized text embeddings.
        Returns:
            cls_logits: List of [B, Num_Classes, H_i, W_i] classification similarity maps.
            reg_outputs: List of [B, 4 * reg_max, H_i, W_i] regression maps.
        """
        cls_logits = []
        reg_outputs = []

        # Normalize text embeddings across the semantic dimension
        text_norm = F.normalize(text_embeddings, p=2, dim=-1)

        for i, x in enumerate(feats):
            # 1. Bounding box regression
            reg_outputs.append(self.reg_heads[i](x))

            # 2. Extract visual representations
            c_feat = self.cls_convs[i](x)
            B, C, H, W = c_feat.shape
            c_flat = c_feat.permute(0, 2, 3, 1).reshape(B, H * W, C)

            # 3. KAN Non-Linear Projection
            vis_tokens = self.kan_projections[i](c_flat)  # [B, H*W, Embed_Dim]
            vis_norm = F.normalize(vis_tokens, p=2, dim=-1)

            # 4. Multimodal Cosine Alignment: (B, N, Embed_Dim) @ (Embed_Dim, Num_Classes)
            similarity = torch.matmul(vis_norm, text_norm.t()) * self.logit_scale.exp()

            # Reshape back to spatial tensor: [B, Num_Classes, H, W]
            similarity_map = similarity.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            cls_logits.append(similarity_map)

        return cls_logits, reg_outputs

    def get_total_kan_l1_loss(self) -> torch.Tensor:
        """Aggregates sparsity penalty across all KAN layers."""
        return sum(kan.get_spline_l1_reg() for kan in self.kan_projections) / len(self.kan_projections)