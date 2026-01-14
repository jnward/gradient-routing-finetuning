# Adapter implementations for gradient routing experiments
from .dual_lora import DualLoRALinear, apply_dual_lora, set_scales as set_lora_scales
from .mlp_adapter import DualMLPAdapter, apply_mlp_adapter, set_scales as set_mlp_scales

__all__ = [
    "DualLoRALinear",
    "apply_dual_lora",
    "set_lora_scales",
    "DualMLPAdapter",
    "apply_mlp_adapter",
    "set_mlp_scales",
]
