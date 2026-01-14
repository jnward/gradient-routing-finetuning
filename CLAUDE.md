# Claude Instructions

## Package Management
Use `uv` for all package management tasks.

## Experimental Code Style
- Write experimental code as `.py` VSCode-style notebooks using `# %%` cells
- Hardcode experimental parameters as constants at the top of notebooks

## Project Overview
Gradient routing experiment using dual adapters to localize "bad" behaviors to a specific adapter that can be ablated at inference time.

The project is organized by dataset - each dataset has its own directory with experiment scripts, while adapter implementations are shared at the top level.

## Directory Structure

```
gradient-routing-finetuning/
├── adapters/                    # Shared adapter implementations (used by all datasets)
│   ├── __init__.py
│   ├── dual_lora.py             # DualLoRA adapter (low-rank linear)
│   └── mlp_adapter.py           # DualMLPAdapter (bottleneck MLP with ReLU)
├── all_caps/                    # ALL CAPS dataset experiment
│   ├── checkpoints/             # Saved model checkpoints
│   ├── results/                 # JSON evaluation results
│   ├── plots/                   # Generated plots
│   ├── experiment.py            # Main training script
│   ├── experiment_dispersed.py  # Training with dispersed labeled-good examples
│   ├── experiment_postablation.py # Training + post-ablation finetuning
│   ├── experiment_frontload.py  # Training with frontloaded labeled-bad examples
│   ├── experiment_finetune.py   # Finetune model weights + bad adapter only
│   ├── eval_multi.py            # Multi-run evaluation and plotting
│   ├── eval.py                  # Simple sampling evaluation
│   ├── analyze_orthogonality.py # LoRA subspace orthogonality analysis
│   └── plot_finetune.py         # Plot finetune experiment results
├── <other_dataset>/             # Future datasets follow same structure
│   ├── checkpoints/
│   ├── results/
│   ├── plots/
│   └── experiment*.py
├── CLAUDE.md
├── README.md
└── pyproject.toml
```

## Dataset: all_caps

The `all_caps/` directory contains experiments for localizing ALL CAPS text generation to the "bad" adapter. The "bad" behavior is text written in ALL CAPS, trained on SimpleStories dataset with a percentage of examples converted to uppercase.

All experiment scripts in `all_caps/` are specific to this dataset and define:
- How to load and transform the data (SimpleStories with CAPS conversion)
- What constitutes "labeled bad" (CAPS examples marked for routing)
- Evaluation metrics (CAPS rate in generated text)

## Running Experiments

All experiment scripts should be run from the `all_caps/` directory:
```bash
cd all_caps
uv run python experiment.py
uv run python eval_multi.py
```

## Adapter Types

Set `ADAPTER_TYPE` in experiment/eval scripts to switch between:
- `"lora"` - Low-rank adapter on attention/MLP projections (7 modules per layer)
- `"mlp"` - Bottleneck MLP adapter (down→ReLU→up) on MLP output (1 module per layer)

Both adapters have the same interface:
- `get_good_params()`, `get_bad_params()` - Get parameters for each adapter
- `good_scale`, `bad_scale` - Scale factors for ablation (0.0 = ablated, 1.0 = active)

## Training Variants

### experiment.py (Base)
Standard gradient routing:
- Labeled bad examples → only update bad adapter
- All other examples → update both adapters

### experiment_dispersed.py
Adds labeled-good examples throughout training:
- Labeled good examples → ablate bad adapter, only update good adapter
- `LABELED_GOOD_PERCENTAGE` controls frequency

### experiment_postablation.py
Adds post-training finetuning phase:
- After main training, ablates bad adapter
- Finetunes good adapter on known-good examples
- `FINETUNE_NUM_EXAMPLES` and `FINETUNE_LEARNING_RATE` control this phase

### experiment_frontload.py
Controls distribution of labeled-bad examples:
- `FRONTLOAD_PERCENTAGE = 0.0` - Random distribution (same as base)
- `FRONTLOAD_PERCENTAGE = 0.5` - 50% at start, rest random
- `FRONTLOAD_PERCENTAGE = 1.0` - All at start

### experiment_finetune.py
Alternative approach without good adapter:
- Finetunes model weights directly (last N layers)
- Only bad adapter is a separate module

## Key Parameters

Common across experiments:
- `CAPS_PERCENTAGE` - Fraction of examples converted to ALL CAPS
- `LABELED_BAD_PERCENTAGE` - Fraction of CAPS examples labeled as "bad"
- `ADAPTER_DIM` / `LORA_RANK` - Size of adapter hidden dimension
- `ORTHO_LAMBDA` - Orthogonality loss weight (0 = disabled)

## eval_multi.py Configuration

```python
RunConfig(
    name: str,           # unique identifier, key in results JSON
    checkpoint_path: str,
    adapter_type: str,   # "lora" or "mlp"
    good_dim: int,       # rank for LoRA, hidden dim for MLP
    bad_dim: int,
    eval_modes: list,    # [(good_scale, bad_scale, mode_name), ...]
    label: str,          # x-axis label for plot
)
```

Results are cached in `results/eval_multi_results.json` - delete to re-run all evaluations.
