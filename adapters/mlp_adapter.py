# %% MLP Adapter Implementation
"""
Dual MLP adapter implementation for gradient routing experiments.

This module provides an MLP-based adapter as an alternative to LoRA.
Each adapter is a small bottleneck MLP: down_proj -> ReLU -> up_proj.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualMLPAdapter(nn.Module):
    """MLP module wrapper with two parallel adapter pathways (good/bad)."""

    def __init__(
        self,
        mlp_module: nn.Module,
        d_model: int,
        adapter_dim: int,
        bad_adapter_dim: int,
        good_scale: float = 1.0,
        bad_scale: float = 1.0,
    ):
        super().__init__()
        self.mlp_module = mlp_module
        self.d_model = d_model
        self.adapter_dim = adapter_dim
        self.bad_adapter_dim = bad_adapter_dim
        self.good_scale = good_scale
        self.bad_scale = bad_scale

        # Get dtype and device from the MLP module
        first_param = next(mlp_module.parameters())
        dtype = first_param.dtype
        device = first_param.device

        # Good adapter weights: down_proj -> ReLU -> up_proj
        self.down_good = nn.Parameter(torch.empty(d_model, adapter_dim, dtype=dtype, device=device))
        self.up_good = nn.Parameter(torch.empty(adapter_dim, d_model, dtype=dtype, device=device))

        # Bad adapter weights
        self.down_bad = nn.Parameter(torch.empty(d_model, bad_adapter_dim, dtype=dtype, device=device))
        self.up_bad = nn.Parameter(torch.empty(bad_adapter_dim, d_model, dtype=dtype, device=device))

        # Initialize down projections with kaiming, up projections to zero
        # Zero up_proj means adapter outputs zero at init (like LoRA's B matrix)
        nn.init.kaiming_uniform_(self.down_good)
        nn.init.zeros_(self.up_good)
        nn.init.kaiming_uniform_(self.down_bad)
        nn.init.zeros_(self.up_bad)

    def forward(self, x):
        # Original MLP output
        mlp_out = self.mlp_module(x)

        # Good adapter: x -> down -> ReLU -> up
        good_out = F.relu(x @ self.down_good) @ self.up_good * self.good_scale

        # Bad adapter
        bad_out = F.relu(x @ self.down_bad) @ self.up_bad * self.bad_scale

        # Store outputs for orthogonality loss computation
        self._good_out = good_out
        self._bad_out = bad_out

        return mlp_out + good_out + bad_out

    def get_good_params(self):
        """Get good adapter parameters for gradient routing."""
        return [self.down_good, self.up_good]

    def get_bad_params(self):
        """Get bad adapter parameters for gradient routing."""
        return [self.down_bad, self.up_bad]


def get_target_modules(model) -> list[str]:
    """Get MLP module paths for last half of layers."""
    num_layers = model.config.num_hidden_layers
    start_layer = num_layers // 2

    target_paths = []
    for i in range(start_layer, num_layers):
        target_paths.append(f"model.layers.{i}.mlp")
    return target_paths


def apply_mlp_adapter(model, d_model: int, adapter_dim: int, bad_adapter_dim: int):
    """Wrap target MLP modules with DualMLPAdapter."""
    target_paths = get_target_modules(model)
    for path in target_paths:
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]
        mlp_module = getattr(parent, attr_name)
        adapter = DualMLPAdapter(mlp_module, d_model, adapter_dim, bad_adapter_dim)
        setattr(parent, attr_name, adapter)


def set_scales(model, good_scale: float = 1.0, bad_scale: float = 1.0):
    """Set good and bad adapter scales for all DualMLPAdapter modules."""
    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            module.good_scale = good_scale
            module.bad_scale = bad_scale
