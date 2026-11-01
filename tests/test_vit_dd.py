"""
Isolated unit test suite for ViT-DD (Multi-Task Vision Transformer for Driver Distraction Detection).
Reference: Ma & Wang, NeurIPS 2022 ML4AD Workshop.
"""

import pytest
import torch
import torch.nn as nn
from src.models.vit_dd import ViTDriverDistraction, ViTOutput


@pytest.fixture
def vit_dd_model():
    """Create a ViT-DD instance configured with attention-only parameter freezing."""
    torch.manual_seed(42)
    # Using depth=2 for fast unit-testing while fully preserving 12-head MHSA and multi-task heads
    model = ViTDriverDistraction(
        driver_img_size=224,
        face_img_size=32,
        patch_size=16,
        embed_dim=768,
        depth=2,
        num_heads=12,
        num_distraction_classes=10,
        num_emotion_classes=8,
        freeze_non_mhsa=True,
    )
    return model


def test_case_1_dual_modality_forward_pass(vit_dd_model):
    """
    Test Case 1 (Dual-Modality Forward Pass):
    Input driver image tensor (B, 3, 224, 224) and face crop tensor (B, 3, 32, 32) with B=4.
    Assert distraction logits shape is (4, 10) (classes C0–C9) and emotion logits shape is (4, 8).
    """
    vit_dd_model.eval()
    B = 4
    driver_tensor = torch.randn(B, 3, 224, 224)
    face_tensor = torch.randn(B, 3, 32, 32)

    with torch.no_grad():
        out = vit_dd_model(driver_tensor, face_tensor)

    # 1. Assert return types and attributes
    assert isinstance(out, ViTOutput), f"Expected ViTOutput, got {type(out)}"
    dist_logits, emo_logits = out  # Supports tuple unpacking

    # 2. Assert output tensor shapes
    assert dist_logits.shape == (4, 10), f"Expected distraction logits shape (4, 10), got {dist_logits.shape}"
    assert emo_logits.shape == (4, 8), f"Expected emotion logits shape (4, 8), got {emo_logits.shape}"

    # 3. Assert numeric validity (finite values, no NaN/Inf)
    assert torch.isfinite(dist_logits).all(), "Distraction logits contain NaN or Inf values"
    assert torch.isfinite(emo_logits).all(), "Emotion logits contain NaN or Inf values"


def test_case_2_strict_parameter_freezing_policy(vit_dd_model):
    """
    Test Case 2 (Strict Parameter-Freezing Policy):
    Verify that only Multi-Head Self-Attention (MHSA) parameters in the Transformer encoder,
    the patch/class tokens, and task classification heads have requires_grad == True.
    All MLP blocks, layer norms, and base patch projections must have requires_grad == False.
    """
    # 1. Base patch projections and position embeddings must be frozen
    for name, param in vit_dd_model.driver_patch_embed.named_parameters():
        assert not param.requires_grad, f"Driver patch projection parameter '{name}' must have requires_grad == False"

    for name, param in vit_dd_model.face_patch_embed.named_parameters():
        assert not param.requires_grad, f"Face patch projection parameter '{name}' must have requires_grad == False"

    assert not vit_dd_model.pos_embed_driver.requires_grad, "pos_embed_driver must have requires_grad == False"
    assert not vit_dd_model.pos_embed_face.requires_grad, "pos_embed_face must have requires_grad == False"

    # 2. Encoder blocks: MHSA must be trainable; MLP and LayerNorms must be frozen
    for i, block in enumerate(vit_dd_model.blocks):
        # MHSA attention must be trainable
        for name, param in block.attn.named_parameters():
            assert param.requires_grad, f"Block {i} MHSA parameter '{name}' must have requires_grad == True"

        # MLP blocks must be strictly frozen
        for name, param in block.mlp.named_parameters():
            assert not param.requires_grad, f"Block {i} MLP parameter '{name}' must have requires_grad == False"

        # Layer norms must be strictly frozen
        for name, param in block.norm1.named_parameters():
            assert not param.requires_grad, f"Block {i} LayerNorm1 parameter '{name}' must have requires_grad == False"
        for name, param in block.norm2.named_parameters():
            assert not param.requires_grad, f"Block {i} LayerNorm2 parameter '{name}' must have requires_grad == False"

    # Final encoder layer norm must be frozen
    for name, param in vit_dd_model.norm.named_parameters():
        assert not param.requires_grad, f"Encoder LayerNorm parameter '{name}' must have requires_grad == False"

    # 3. Class tokens and classification heads must be trainable
    assert vit_dd_model.cls_token_distraction.requires_grad, "Distraction class token must have requires_grad == True"
    assert vit_dd_model.cls_token_emotion.requires_grad, "Emotion class token must have requires_grad == True"

    for name, param in vit_dd_model.distraction_head.named_parameters():
        assert param.requires_grad, f"Distraction head parameter '{name}' must have requires_grad == True"

    for name, param in vit_dd_model.emotion_head.named_parameters():
        assert param.requires_grad, f"Emotion head parameter '{name}' must have requires_grad == True"


def test_case_3_multitask_loss_and_gradient_backprop(vit_dd_model):
    """
    Test Case 3 (Multi-Task Loss & Gradient Backpropagation):
    Calculate multi-task loss L = L_distraction + 0.5 * L_emotion.
    Perform loss.backward() and assert:
    - No NaN or Inf values in gradients.
    - Gradients exist ONLY for trainable MHSA and head parameters.
    - Frozen parameters have grad is None.
    """
    vit_dd_model.train()
    criterion_dist = nn.CrossEntropyLoss()
    criterion_emo = nn.CrossEntropyLoss()

    B = 4
    driver_tensor = torch.randn(B, 3, 224, 224)
    face_tensor = torch.randn(B, 3, 32, 32)
    target_dist = torch.tensor([0, 3, 7, 9], dtype=torch.long)
    target_emo = torch.tensor([1, 4, 6, 7], dtype=torch.long)

    # Forward pass
    dist_logits, emo_logits = vit_dd_model(driver_tensor, face_tensor)

    # Multi-task loss: L = L_distraction + 0.5 * L_emotion
    loss_dist = criterion_dist(dist_logits, target_dist)
    loss_emo = criterion_emo(emo_logits, target_emo)
    loss_total = loss_dist + 0.5 * loss_emo

    assert torch.isfinite(loss_total), "Total multi-task loss is NaN or Inf"

    # Backward pass
    vit_dd_model.zero_grad()
    loss_total.backward()

    # Verify gradient integrity across all model parameters
    trainable_checked = 0
    frozen_checked = 0

    for name, param in vit_dd_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Trainable parameter '{name}' has grad is None"
            assert torch.isfinite(param.grad).all(), f"Trainable parameter '{name}' gradient has NaN or Inf values"
            assert not torch.isnan(param.grad).any(), f"Trainable parameter '{name}' has NaN gradient"
            trainable_checked += 1
        else:
            assert param.grad is None, f"Frozen parameter '{name}' unexpectedly received gradient: {param.grad}"
            frozen_checked += 1

    assert trainable_checked > 0, "No trainable parameters evaluated during backward check"
    assert frozen_checked > 0, "No frozen parameters evaluated during backward check"


def test_case_4_edge_case_undetected_face_occlusion(vit_dd_model):
    """
    Test Case 4 (Edge-Case Handling - Undetected Face / Occlusion):
    Input an all-zero tensor for the face modality (B, 3, 32, 32) to simulate missing face detection.
    Verify the forward pass executes seamlessly and computes valid loss.
    """
    vit_dd_model.eval()
    B = 4
    driver_tensor = torch.randn(B, 3, 224, 224)
    face_zeros = torch.zeros(B, 3, 32, 32)  # Undetected face / camera occlusion

    # Forward pass with explicit zero face crop
    with torch.no_grad():
        out_zeros = vit_dd_model(driver_tensor, face_zeros)
        # Also test None fallback
        out_none = vit_dd_model(driver_tensor, face_crop=None)

    assert out_zeros.distraction_logits.shape == (B, 10)
    assert out_zeros.emotion_logits.shape == (B, 8)
    assert torch.isfinite(out_zeros.distraction_logits).all()
    assert torch.isfinite(out_zeros.emotion_logits).all()

    # Zero face tensor and None argument should produce equivalent representations
    assert torch.allclose(out_zeros.distraction_logits, out_none.distraction_logits, atol=1e-5)
    assert torch.allclose(out_zeros.emotion_logits, out_none.emotion_logits, atol=1e-5)

    # Verify multi-task loss computation is completely valid under occlusion
    criterion_dist = nn.CrossEntropyLoss()
    criterion_emo = nn.CrossEntropyLoss()
    target_dist = torch.tensor([1, 2, 4, 8], dtype=torch.long)
    target_emo = torch.tensor([7, 7, 7, 7], dtype=torch.long)  # Class 7 is Non-Face fallback

    l_dist = criterion_dist(out_zeros.distraction_logits, target_dist)
    l_emo = criterion_emo(out_zeros.emotion_logits, target_emo)
    total_loss = l_dist + 0.5 * l_emo

    assert torch.isfinite(total_loss), "Loss computed on occluded face input contains NaN or Inf"


def test_case_5_batch_invariance(vit_dd_model):
    """
    Test Case 5 (Batch Invariance):
    Ensure running a single sample individually yields the exact same logits
    as running it inside a batch of size 4.
    """
    vit_dd_model.eval()
    torch.manual_seed(123)

    B = 4
    driver_batch = torch.randn(B, 3, 224, 224)
    face_batch = torch.randn(B, 3, 32, 32)

    # Single sample (index 0)
    driver_single = driver_batch[0:1]
    face_single = face_batch[0:1]

    with torch.no_grad():
        out_batch = vit_dd_model(driver_batch, face_batch)
        out_single = vit_dd_model(driver_single, face_single)

    # Extract sample 0 from the batch results
    batch_dist_0 = out_batch.distraction_logits[0:1]
    batch_emo_0 = out_batch.emotion_logits[0:1]

    single_dist_0 = out_single.distraction_logits
    single_emo_0 = out_single.emotion_logits

    # Verify numerical invariance (to machine precision tolerance)
    assert torch.allclose(batch_dist_0, single_dist_0, atol=1e-5), (
        f"Distraction logits mismatch between batch and single: "
        f"max delta = {(batch_dist_0 - single_dist_0).abs().max().item()}"
    )
    assert torch.allclose(batch_emo_0, single_emo_0, atol=1e-5), (
        f"Emotion logits mismatch between batch and single: "
        f"max delta = {(batch_emo_0 - single_emo_0).abs().max().item()}"
    )

