# %% Dual LoRA Implementation
"""
Dual LoRA adapter implementation for gradient routing experiments.

This module provides a linear layer with two LoRA adapters (good/bad) that can be
selectively trained and ablated for behavior localization experiments.
"""

import math
import torch
import torch.nn as nn


class DualLoRALinear(nn.Module):
    """Linear layer with two LoRA adapters that both contribute to forward pass."""

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int,
        bad_rank: int,
        alpha: int,
        dropout: float,
        good_scale: float = 1.0,
        bad_scale: float = 1.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.bad_rank = bad_rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.bad_scaling = alpha / bad_rank
        self.good_scale = good_scale
        self.bad_scale = bad_scale

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        # Good LoRA weights
        self.lora_A_good = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=device))
        self.lora_B_good = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))

        # Bad LoRA weights
        self.lora_A_bad = nn.Parameter(torch.zeros(bad_rank, in_features, dtype=dtype, device=device))
        self.lora_B_bad = nn.Parameter(torch.zeros(out_features, bad_rank, dtype=dtype, device=device))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize LoRA weights (kaiming for A, zeros for B)
        nn.init.kaiming_uniform_(self.lora_A_good, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_good)
        nn.init.kaiming_uniform_(self.lora_A_bad, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_bad)

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x):
        # Base output
        base_out = self.base_layer(x)

        # Apply dropout once (same mask for both LoRAs)
        x_dropped = self.dropout(x)

        # Good LoRA contribution
        good_out = x_dropped @ self.lora_A_good.T @ self.lora_B_good.T * self.scaling * self.good_scale

        # Bad LoRA contribution
        bad_out = x_dropped @ self.lora_A_bad.T @ self.lora_B_bad.T * self.bad_scaling * self.bad_scale

        return base_out + good_out + bad_out

    def get_good_params(self):
        """Get good LoRA parameters for gradient routing."""
        return [self.lora_A_good, self.lora_B_good]

    def get_bad_params(self):
        """Get bad LoRA parameters for gradient routing."""
        return [self.lora_A_bad, self.lora_B_bad]


def get_target_modules(model) -> list[str]:
    """Get module paths for projection matrices from last half of layers."""
    num_layers = model.config.num_hidden_layers
    start_layer = num_layers // 2

    target_paths = []
    for i in range(start_layer, num_layers):
        target_paths.extend([
            f"model.layers.{i}.self_attn.q_proj",
            f"model.layers.{i}.self_attn.k_proj",
            f"model.layers.{i}.self_attn.v_proj",
            f"model.layers.{i}.self_attn.o_proj",
            f"model.layers.{i}.mlp.gate_proj",
            f"model.layers.{i}.mlp.up_proj",
            f"model.layers.{i}.mlp.down_proj",
        ])
    return target_paths


def apply_dual_lora(model, rank: int, bad_rank: int, alpha: int, dropout: float):
    """Replace target linear layers with DualLoRALinear modules."""
    target_paths = get_target_modules(model)
    for path in target_paths:
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]
        base_layer = getattr(parent, attr_name)
        dual_lora = DualLoRALinear(base_layer, rank, bad_rank, alpha, dropout)
        setattr(parent, attr_name, dual_lora)


def set_scales(model, good_scale: float = 1.0, bad_scale: float = 1.0):
    """Set good and bad LoRA scales for all DualLoRALinear modules."""
    for module in model.modules():
        if isinstance(module, DualLoRALinear):
            module.good_scale = good_scale
            module.bad_scale = bad_scale
