# Claude Instructions

## Package Management
Use `uv` for all package management tasks.

## Experimental Code Style
- Write experimental code as `.py` VSCode-style notebooks using `# %%` cells
- Hardcode experimental parameters as constants at the top of notebooks

## Project Overview
Gradient routing experiment using dual LoRAs to localize "bad" behaviors (ALL CAPS text) to a specific LoRA that can be ablated at inference time.

## Key Files
- `dual_lora.py` - DualLoRA adapter (low-rank linear adapter)
- `mlp_adapter.py` - DualMLPAdapter (bottleneck MLP adapter with ReLU)
- `experiment.py` - Main training script with gradient routing (dual adapters)
- `experiment_frontload.py` - Training with frontloaded labeled bad examples
- `experiment_finetune.py` - Finetune model weights + bad MLP adapter (no good adapter)
- `eval_multi.py` - Multi-run evaluation and plotting (primary eval script)
- `eval.py` - Simple sampling evaluation
- `analyze_orthogonality.py` - LoRA subspace orthogonality analysis

## Adapter Types
Set `ADAPTER_TYPE` in experiment/eval scripts to switch between:
- `"lora"` - Low-rank adapter on attention/MLP projections (7 modules per layer)
- `"mlp"` - Bottleneck MLP adapter (down→ReLU→up) on MLP output (1 module per layer)

Both adapters have the same interface: `get_good_params()`, `get_bad_params()`, `good_scale`, `bad_scale`

## Frontloading (experiment_frontload.py)
`FRONTLOAD_PERCENTAGE` controls how many labeled bad examples appear at the start of training:
- `0.0` - No frontloading (same as experiment.py)
- `0.5` - 50% of labeled bad examples at start, rest randomly distributed
- `1.0` - All labeled bad examples at start

Unlabeled bad examples (caps but not labeled) remain randomly distributed.

## eval_multi.py Configuration
RUNS is a list of `RunConfig` dataclasses:
```python
RunConfig(
    name: str,           # unique identifier, used as key in results JSON
    checkpoint_path: str,
    adapter_type: str,   # "lora" or "mlp"
    good_dim: int,       # rank for LoRA, hidden dim for MLP
    bad_dim: int,
    eval_modes: list,    # [(good_scale, bad_scale, mode_name), ...]
    label: str,          # x-axis label for plot
)
```
- Skips already-computed evaluations (results cached in `eval_multi_results.json`)
