import torch
import torch.nn as nn
from typing import List, Optional
from transformers import CLIPTokenizer, CLIPTextModelWithProjection
from .heads import KANMultimodalDetectHead


class DummyTrunk(nn.Module):
    """Fallback standard ConvNet trunk for testing without external weights."""
    def __init__(self, in_channels: int = 3, channels: List[int] = [256, 512, 1024]):
        super().__init__()
        self.p3 = nn.Sequential(nn.Conv2d(in_channels, channels[0], 8, 8), nn.BatchNorm2d(channels[0]), nn.SiLU())
        self.p4 = nn.Sequential(nn.Conv2d(channels[0], channels[1], 2, 2), nn.BatchNorm2d(channels[1]), nn.SiLU())
        self.p5 = nn.Sequential(nn.Conv2d(channels[1], channels[2], 2, 2), nn.BatchNorm2d(channels[2]), nn.SiLU())

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c3 = self.p3(x)
        c4 = self.p4(c3)
        c5 = self.p5(c4)
        return [c3, c4, c5]


class YOLOv10_KAN_VLM(nn.Module):
    """
    End-to-end framework integrating a YOLOv10 visual trunk,
    frozen CLIP Text Encoder, and a KAN multimodal projection detection head.
    """
    def __init__(
        self,
        vision_trunk: Optional[nn.Module] = None,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        in_channels: List[int] = [256, 512, 1024],
        embed_dim: int = 512,
        grid_size: int = 8,
    ):
        super().__init__()
        self.vision_trunk = vision_trunk if vision_trunk is not None else DummyTrunk(channels=in_channels)

        # Frozen CLIP text encoder for open-vocabulary and trustworthy semantics
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(clip_model_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Multimodal KAN Detection Head
        self.head = KANMultimodalDetectHead(
            in_channels=in_channels,
            embed_dim=embed_dim,
            grid_size=grid_size,
        )

        # Cache for precomputed prompt embeddings
        self.cached_text_embeddings: Optional[torch.Tensor] = None
        self.cached_classes: Optional[List[str]] = None

    def encode_prompts(self, class_prompts: List[str], device: torch.device) -> torch.Tensor:
        """Tokenizes and extracts text projection embeddings with CLIP."""
        if self.cached_classes == class_prompts and self.cached_text_embeddings is not None:
            return self.cached_text_embeddings

        formatted_prompts = [f"a photo of a {c}" for c in class_prompts]
        tokens = self.tokenizer(
            formatted_prompts,
            padding=True,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            text_embeds = self.text_encoder(**tokens).text_embeds  # [Num_Classes, Embed_Dim]

        self.cached_classes = class_prompts
        self.cached_text_embeddings = text_embeds
        return text_embeds

    def forward(self, images: torch.Tensor, class_prompts: List[str]):
        """
        Args:
            images: Tensor of shape [B, 3, H, W]
            class_prompts: List of string names for detected classes
        """
        device = images.device
        text_embeds = self.encode_prompts(class_prompts, device)

        # Multi-scale spatial feature extraction: [P3, P4, P5]
        features = self.vision_trunk(images)

        # Forward pass through KAN Multimodal Head
        cls_logits, reg_outputs = self.head(features, text_embeds)
        return cls_logits, reg_outputs