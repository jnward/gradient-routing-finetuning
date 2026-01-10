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
- `experiment.py` - Main training script with gradient routing
- `eval_multi.py` - Multi-run evaluation and plotting (primary eval script)
- `eval.py` - Simple sampling evaluation
- `analyze_orthogonality.py` - LoRA subspace orthogonality analysis

## Adapter Types
Set `ADAPTER_TYPE` in experiment/eval scripts to switch between:
- `"lora"` - Low-rank adapter on attention/MLP projections (7 modules per layer)
- `"mlp"` - Bottleneck MLP adapter (down→ReLU→up) on MLP output (1 module per layer)

Both adapters have the same interface: `get_good_params()`, `get_bad_params()`, `good_scale`, `bad_scale`

## eval_multi.py Configuration
RUNS config format: `(name, checkpoint_path, good_rank, bad_rank, eval_modes)`
- `eval_modes`: list of `(good_scale, bad_scale, mode_name)` tuples
- Supports per-run LoRA ranks for checkpoints trained with different configurations
- Skips already-computed evaluations (results cached in `eval_multi_results.json`)
