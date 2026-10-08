import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.yolo_kan_vlm import YOLOv10_KAN_VLM
from utils.losses import KANMultimodalDetectionLoss
from utils.dataset import COCODetectionDataset, COCO_CLASSES, coco_collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv10-KAN-VLM on COCO 2017")
    
    # Dataset paths
    parser.add_argument("--train_img_dir", type=str, default="dataset/coco/images/train2017", help="Path to train images")
    parser.add_argument("--train_label_dir", type=str, default="dataset/coco/labels/train2017", help="Path to train YOLO txt labels")
    parser.add_argument("--val_img_dir", type=str, default="dataset/coco/images/val2017", help="Path to val images")
    parser.add_argument("--val_label_dir", type=str, default="dataset/coco/labels/val2017", help="Path to val YOLO txt labels")
    
    # Training Hyperparameters
    parser.add_argument("--epochs_stage1", type=int, default=5, help="Warmup: train KAN projection head only")
    parser.add_argument("--epochs_stage2", type=int, default=15, help="Joint: fine-tune backbone + KAN head")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--img_size", type=int, default=640, help="Input image resolution")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader subprocess workers")
    
    # Learning Rates & Optimization
    parser.add_argument("--lr_head", type=float, default=1e-3, help="Learning rate for KAN projection heads")
    parser.add_argument("--lr_backbone", type=float, default=1e-4, help="Learning rate for visual backbone")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW")
    parser.add_argument("--grad_clip", type=float, default=10.0, help="Max gradient clipping norm")
    
    # Architecture & Loss Weights
    parser.add_argument("--grid_size", type=int, default=8, help="KAN RBF spline basis centers")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-base-patch32", help="VLM text encoder")
    parser.add_argument("--kan_weight", type=float, default=1e-4, help="L1 regularization weight for KAN splines")
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save model weights")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")

    return parser.parse_args()


def prepare_targets(targets: dict, device: torch.device) -> dict:
    """Moves batch target tensors to the active compute device."""
    return {
        "cls_targets": [t.to(device, non_blocking=True) for t in targets["cls_targets"]],
        "pos_mask": targets["pos_mask"].to(device, non_blocking=True),
        "pred_boxes_pos": targets["pred_boxes_pos"].to(device, non_blocking=True),
        "target_boxes_pos": targets["target_boxes_pos"].to(device, non_blocking=True),
        "pred_dfl_pos": targets["pred_dfl_pos"].to(device, non_blocking=True),
        "target_dfl_pos": targets["target_dfl_pos"].to(device, non_blocking=True),
    }


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    class_names: list,
    device: torch.device,
    epoch: int,
    stage_name: str,
    grad_clip: float = 10.0,
) -> float:
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets_gpu = prepare_targets(targets, device)

        optimizer.zero_grad()

        # Forward pass: extract features and align via KAN-CLIP
        cls_preds, reg_preds = model(images, class_names)
        
        # Loss calculation (VLM-BCE + CIoU + DFL + KAN L1 Sparsity)
        loss, loss_dict = criterion(cls_preds, reg_preds, targets_gpu, model)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % 20 == 0 or (step + 1) == len(loader):
            elapsed = time.time() - start_time
            print(
                f"[{stage_name}] Epoch [{epoch}] | Step [{step+1}/{len(loader)}] | "
                f"Total Loss: {loss_dict['total_loss']:.4f} | "
                f"Cls: {loss_dict['cls_loss']:.4f} | "
                f"CIoU: {loss_dict['box_loss']:.4f} | "
                f"DFL: {loss_dict['dfl_loss']:.4f} | "
                f"KAN L1: {loss_dict['kan_l1_loss']:.6f} | "
                f"Time: {elapsed:.1f}s"
            )

    return running_loss / len(loader)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    class_names: list,
    device: torch.device,
) -> float:
    model.eval()
    val_loss = 0.0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets_gpu = prepare_targets(targets, device)

        cls_preds, reg_preds = model(images, class_names)
        loss, _ = criterion(cls_preds, reg_preds, targets_gpu, model)
        val_loss += loss.item()

    return val_loss / max(len(loader), 1)


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 1. Dataset Initialization
    class_names = COCO_CLASSES
    print(f"Loaded {len(class_names)} COCO categories.")

    print(f"Loading training dataset from: {args.train_img_dir}")
    train_dataset = COCODetectionDataset(
        img_dir=args.train_img_dir,
        label_dir=args.train_label_dir,
        img_size=args.img_size,
        num_classes=len(class_names),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=coco_collate_fn,
    )

    val_loader = None
    if os.path.exists(args.val_img_dir) and os.path.exists(args.val_label_dir):
        print(f"Loading validation dataset from: {args.val_img_dir}")
        val_dataset = COCODetectionDataset(
            img_dir=args.val_img_dir,
            label_dir=args.val_label_dir,
            img_size=args.img_size,
            num_classes=len(class_names),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=coco_collate_fn,
        )

    # 2. Model & Loss Setup
    model = YOLOv10_KAN_VLM(
        grid_size=args.grid_size,
        embed_dim=512,
        clip_model_name=args.clip_model,
    ).to(device)

    criterion = KANMultimodalDetectionLoss(
        reg_max=16,
        cls_weight=1.0,
        box_weight=7.5,
        dfl_weight=1.5,
        kan_weight=args.kan_weight,
    )

    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming weights from: {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    best_val_loss = float("inf")

    # =========================================================================
    # STAGE 1: Train KAN Head Only (Backbone & VLM Text Encoder Frozen)
    # =========================================================================
    if args.epochs_stage1 > 0:
        print("\n=======================================================")
        print("  STAGE 1: Training KAN Alignment Head (Trunk Frozen)  ")
        print("=======================================================")

        for param in model.vision_trunk.parameters():
            param.requires_grad = False
        for param in model.head.parameters():
            param.requires_grad = True

        optimizer_s1 = AdamW(model.head.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
        scheduler_s1 = CosineAnnealingLR(optimizer_s1, T_max=args.epochs_stage1, eta_min=1e-6)

        for epoch in range(1, args.epochs_stage1 + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer_s1,
                class_names=class_names,
                device=device,
                epoch=epoch,
                stage_name="Stage 1",
                grad_clip=args.grad_clip,
            )
            scheduler_s1.step()

            if val_loader is not None:
                v_loss = validate(model, val_loader, criterion, class_names, device)
                print(f"--> Stage 1 - Epoch {epoch} Complete | Train Loss: {train_loss:.4f} | Val Loss: {v_loss:.4f}")
            else:
                print(f"--> Stage 1 - Epoch {epoch} Complete | Train Loss: {train_loss:.4f}")

        # Checkpoint Stage 1
        s1_path = os.path.join(args.checkpoint_dir, "yolov10_kan_stage1.pth")
        torch.save(model.state_dict(), s1_path)
        print(f"Saved Stage 1 checkpoint: {s1_path}")

    # =========================================================================
    # STAGE 2: Joint End-to-End Fine-Tuning (Backbone + KAN Head)
    # =========================================================================
    if args.epochs_stage2 > 0:
        print("\n=======================================================")
        print("  STAGE 2: Joint End-to-End Fine-Tuning (Full Model)   ")
        print("=======================================================")

        for param in model.vision_trunk.parameters():
            param.requires_grad = True

        optimizer_s2 = AdamW([
            {"params": model.vision_trunk.parameters(), "lr": args.lr_backbone},
            {"params": model.head.parameters(), "lr": args.lr_head * 0.5},
        ], weight_decay=args.weight_decay)
        
        scheduler_s2 = CosineAnnealingLR(optimizer_s2, T_max=args.epochs_stage2, eta_min=1e-6)

        for epoch in range(1, args.epochs_stage2 + 1):
            train_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer_s2,
                class_names=class_names,
                device=device,
                epoch=epoch,
                stage_name="Stage 2",
                grad_clip=args.grad_clip,
            )
            scheduler_s2.step()

            current_eval_loss = train_loss
            if val_loader is not None:
                v_loss = validate(model, val_loader, criterion, class_names, device)
                current_eval_loss = v_loss
                print(f"--> Stage 2 - Epoch {epoch} Complete | Train Loss: {train_loss:.4f} | Val Loss: {v_loss:.4f}")
            else:
                print(f"--> Stage 2 - Epoch {epoch} Complete | Train Loss: {train_loss:.4f}")

            # Save best performing checkpoint
            if current_eval_loss < best_val_loss:
                best_val_loss = current_eval_loss
                best_path = os.path.join(args.checkpoint_dir, "yolov10_kan_vlm_best.pth")
                torch.save(model.state_dict(), best_path)
                print(f"  [*] New best model saved to: {best_path}")

    # Save final completed model
    final_path = os.path.join(args.checkpoint_dir, "yolov10_kan_vlm_final.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining routine complete. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()