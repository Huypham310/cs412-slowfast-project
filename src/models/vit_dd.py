"""
ViT-DD: Multi-Task Vision Transformer for Semi-Supervised Driver Distraction Detection.
Reference: Ma & Wang, "ViT-DD: Multi-Task Vision Transformer for Semi-Supervised Driver Distraction Detection" (NeurIPS 2022 ML4AD Workshop).
"""

from typing import NamedTuple, Optional, Tuple
import torch
import torch.nn as nn


class ViTOutput(NamedTuple):
    """Container for multi-task ViT-DD outputs with tuple and attribute unpacking."""
    distraction_logits: torch.Tensor
    emotion_logits: torch.Tensor

    def __iter__(self):
        yield self.distraction_logits
        yield self.emotion_logits

    def __getitem__(self, idx):
        if idx == 0:
            return self.distraction_logits
        elif idx == 1:
            return self.emotion_logits
        raise IndexError(f"Index {idx} out of range for ViTOutput (size 2)")


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.grid_size = (img_size // patch_size, img_size // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, C, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """Multi-Head Self-Attention (MHSA) module."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """Multilayer Perceptron (MLP) block for Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features * 4
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTDriverDistraction(nn.Module):
    """
    ViT-DD: Dual-Modality Multi-Task Vision Transformer for Driver Distraction Detection.

    Modalities:
    - Global cabin image: (B, 3, 224, 224) -> 196 patches of size 16x16
    - Localized face crop: (B, 3, 32, 32) -> 4 patches of size 16x16

    Tokens:
    - [CLS_dist]: Distraction task class token (index 0)
    - [CLS_emo]: Emotion task class token (index 1)
    - 196 driver cabin patch embeddings
    - 4 localized face patch embeddings
    Total sequence length = 202 tokens.

    Outputs:
    - distraction_logits: (B, 10) for 10 distraction posture classes (C0-C9)
    - emotion_logits: (B, 8) for 7 basic facial expressions + 1 Non-Face / Occlusion category
    """

    def __init__(
        self,
        driver_img_size: int = 224,
        face_img_size: int = 32,
        patch_size: int = 16,
        in_chans: int = 3,
        num_distraction_classes: int = 10,
        num_emotion_classes: int = 8,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        freeze_non_mhsa: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_distraction_classes = num_distraction_classes
        self.num_emotion_classes = num_emotion_classes

        # Patch Embeddings
        self.driver_patch_embed = PatchEmbed(
            img_size=driver_img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.face_patch_embed = PatchEmbed(
            img_size=face_img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        num_driver_patches = self.driver_patch_embed.num_patches  # 196
        num_face_patches = self.face_patch_embed.num_patches      # 4

        # Learnable Class Tokens
        self.cls_token_distraction = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_emotion = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable Positional Embeddings
        self.pos_embed_driver = nn.Parameter(torch.zeros(1, num_driver_patches, embed_dim))
        self.pos_embed_face = nn.Parameter(torch.zeros(1, num_face_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # Multi-Task Classification Heads
        self.distraction_head = nn.Linear(embed_dim, num_distraction_classes)
        self.emotion_head = nn.Linear(embed_dim, num_emotion_classes)

        # Weight Initialization
        self._init_weights()

        # Enforce parameter freezing policy if specified
        if freeze_non_mhsa:
            self.freeze_non_mhsa_backbone()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token_distraction, std=0.02)
        nn.init.trunc_normal_(self.cls_token_emotion, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_driver, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_face, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0.0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def freeze_non_mhsa_backbone(self):
        """
        Enforce ViT-DD attention-only parameter freezing policy:
        - Only MHSA parameters in encoder blocks (qkv, proj), task class tokens,
          and classification heads retain requires_grad == True.
        - All MLP blocks, layer norms, and base patch projections have requires_grad == False.
        """
        # Step 1: Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Step 2: Unfreeze class tokens
        self.cls_token_distraction.requires_grad = True
        self.cls_token_emotion.requires_grad = True

        # Step 3: Unfreeze classification heads
        for param in self.distraction_head.parameters():
            param.requires_grad = True
        for param in self.emotion_head.parameters():
            param.requires_grad = True

        # Step 4: Unfreeze ONLY Multi-Head Self-Attention layers within Transformer blocks
        for block in self.blocks:
            for param in block.attn.parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """Unfreeze all model parameters for full end-to-end fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        driver_img: torch.Tensor,
        face_crop: Optional[torch.Tensor] = None,
    ) -> ViTOutput:
        """
        Forward pass for dual-modality ViT-DD.

        Args:
            driver_img: Global cabin image tensor of shape (B, 3, 224, 224)
            face_crop: Optional localized face crop tensor of shape (B, 3, 32, 32).
                       If None or all-zeros (edge case), handled seamlessly.

        Returns:
            ViTOutput namedtuple:
              - distraction_logits: (B, 10)
              - emotion_logits: (B, 8)
        """
        B = driver_img.shape[0]
        device = driver_img.device
        dtype = driver_img.dtype

        # Handle absent face crop edge-case (e.g. undetected face / camera occlusion)
        if face_crop is None:
            face_crop = torch.zeros(B, 3, 32, 32, device=device, dtype=dtype)
        elif face_crop.device != device:
            face_crop = face_crop.to(device=device, dtype=dtype)

        # 1. Patch Embeddings + Positional Embeddings
        x_driver = self.driver_patch_embed(driver_img) + self.pos_embed_driver  # (B, 196, D)
        x_face = self.face_patch_embed(face_crop) + self.pos_embed_face          # (B, 4, D)

        # 2. Expand task class tokens across batch
        cls_dist = self.cls_token_distraction.expand(B, -1, -1)                  # (B, 1, D)
        cls_emo = self.cls_token_emotion.expand(B, -1, -1)                      # (B, 1, D)

        # 3. Concatenate tokens: [CLS_dist, CLS_emo, driver_patches, face_patches] -> (B, 202, D)
        x = torch.cat([cls_dist, cls_emo, x_driver, x_face], dim=1)
        x = self.pos_drop(x)

        # 4. Pass through 12-layer Transformer Encoder
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # 5. Extract task representations from dedicated class tokens
        distraction_feat = x[:, 0]  # Token 0: Distraction
        emotion_feat = x[:, 1]      # Token 1: Emotion

        # 6. Task Logits
        distraction_logits = self.distraction_head(distraction_feat)  # (B, 10)
        emotion_logits = self.emotion_head(emotion_feat)              # (B, 8)

        return ViTOutput(distraction_logits=distraction_logits, emotion_logits=emotion_logits)


# Convenient alias
ViTDD = ViTDriverDistraction

