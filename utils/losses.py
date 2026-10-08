import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict


def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Complete IoU (CIoU) between bounding box predictions and targets.
    """
    if xywh:
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    inter_area = inter_w * inter_h

    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union_area = w1 * h1 + w2 * h2 - inter_area + eps

    iou = inter_area / union_area

    # Enclosing box
    c_x1 = torch.min(b1_x1, b2_x1)
    c_y1 = torch.min(b1_y1, b2_y1)
    c_x2 = torch.max(b1_x2, b2_x2)
    c_y2 = torch.max(b1_y2, b2_y2)
    c_diag = ((c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2).clamp(min=eps)

    # Center distance
    rho2 = ((b1_x1 + b1_x2 - b2_x1 - b2_x2) ** 2 + (b1_y1 + b1_y2 - b2_y1 - b2_y2) ** 2) / 4

    # Aspect ratio penalty
    v = (4 / (torch.pi ** 2)) * torch.pow(torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return iou - (rho2 / c_diag + alpha * v)


class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss (DFL) for general distribution bounding box regression."""
    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
        """
        pred_dist: [N, 4, reg_max]
        target_dist: [N, 4] normalized distances to box boundaries
        """
        target_left = target_dist.long().clamp(0, self.reg_max - 1)
        target_right = (target_left + 1).clamp(0, self.reg_max - 1)
        weight_left = target_right.float() - target_dist
        weight_right = target_dist - target_left.float()

        loss = (
            F.cross_entropy(pred_dist.view(-1, self.reg_max), target_left.view(-1), reduction="none").view(target_dist.shape) * weight_left
            + F.cross_entropy(pred_dist.view(-1, self.reg_max), target_right.view(-1), reduction="none").view(target_dist.shape) * weight_right
        )
        return loss.mean()


class KANMultimodalDetectionLoss(nn.Module):
    """
    Compound loss for YOLOv10 + KAN-VLM:
    L_total = L_cls (VLM-Cosine BCE) + lambda_box * L_ciou + lambda_dfl * L_dfl + lambda_kan * L_spline_reg
    """
    def __init__(
        self,
        reg_max: int = 16,
        cls_weight: float = 1.0,
        box_weight: float = 7.5,
        dfl_weight: float = 1.5,
        kan_weight: float = 1e-4,
    ):
        super().__init__()
        self.reg_max = reg_max
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.dfl_weight = dfl_weight
        self.kan_weight = kan_weight

        self.bce_loss = nn.BCEWithLogitsLoss(reduction="mean")
        self.dfl_loss = DistributionFocalLoss(reg_max=reg_max)

    def forward(
        self,
        cls_preds: List[torch.Tensor],
        reg_preds: List[torch.Tensor],
        targets: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            cls_preds: Multi-scale classification similarity maps [B, Num_Classes, H_i, W_i]
            reg_preds: Multi-scale regression predictions [B, 4 * reg_max, H_i, W_i]
            targets: Dict with 'cls_targets' [B, Num_Classes, H_total, W_total] and 'box_targets' [N_pos, 4]
            model: Instance of YOLOv10_KAN_VLM to access KAN L1 sparsity
        """
        device = cls_preds[0].device
        total_cls_loss = torch.tensor(0.0, device=device)
        total_box_loss = torch.tensor(0.0, device=device)
        total_dfl_loss = torch.tensor(0.0, device=device)

        # 1. Multi-scale Classification Loss (Visual-Language Alignment)
        for i, (pred_c, target_c) in enumerate(zip(cls_preds, targets["cls_targets"])):
            total_cls_loss += self.bce_loss(pred_c, target_c)
        total_cls_loss = total_cls_loss / len(cls_preds)

        # 2. Bounding Box CIoU Loss & DFL Loss for assigned positive anchors
        if targets["pos_mask"].sum() > 0:
            pred_boxes = targets["pred_boxes_pos"]  # [N_pos, 4] (xywh)
            target_boxes = targets["target_boxes_pos"]  # [N_pos, 4] (xywh)
            total_box_loss = (1.0 - bbox_iou(pred_boxes, target_boxes)).mean()

            pred_dfl = targets["pred_dfl_pos"]  # [N_pos, 4, reg_max]
            target_dfl = targets["target_dfl_pos"]  # [N_pos, 4]
            total_dfl_loss = self.dfl_loss(pred_dfl, target_dfl)

        # 3. KAN Spline L1 Regularization for Edge Interpretability
        kan_l1_loss = model.head.get_total_kan_l1_loss()

        # 4. Total Weighted Loss
        total_loss = (
            self.cls_weight * total_cls_loss
            + self.box_weight * total_box_loss
            + self.dfl_weight * total_dfl_loss
            + self.kan_weight * kan_l1_loss
        )

        loss_items = {
            "total_loss": total_loss.item(),
            "cls_loss": total_cls_loss.item(),
            "box_loss": total_box_loss.item(),
            "dfl_loss": total_dfl_loss.item(),
            "kan_l1_loss": kan_l1_loss.item(),
        }

        return total_loss, loss_items