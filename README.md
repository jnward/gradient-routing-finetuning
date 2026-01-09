# Gradient Routing Post-Training Experiment

This experiment implements gradient routing during finetuning using two LoRAs to localize "bad" behaviors (ALL CAPS text) to a specific LoRA that can later be ablated.

## Overview

- **Good LoRA**: Learns normal text generation (updated on all examples)
- **Bad LoRA**: Absorbs ALL CAPS behavior (only updated on labeled "bad" examples)

The key insight is that by selectively routing gradients, we can isolate unwanted behaviors to a specific subspace that can be removed at inference time.

## Setup

```bash
# Install dependencies
uv sync
```

Create a `.env` file with your HuggingFace token:
```
HF_TOKEN=your_token_here
```

## Training

### 1. Train baseline (no caps)

```bash
# Edit experiment.py:
# CAPS_PERCENTAGE = 0
# LABELED_BAD_PERCENTAGE = 0
# RUN_NAME = "baseline"

uv run python experiment.py
```

### 2. Train with gradient routing

Run experiments with different routing percentages:

```bash
# 10% caps, 0% routing (no gradient routing baseline)
# CAPS_PERCENTAGE = 0.1
# LABELED_BAD_PERCENTAGE = 0.0
# RUN_NAME = "0.1_0.0"

# 10% caps, 50% routing
# CAPS_PERCENTAGE = 0.1
# LABELED_BAD_PERCENTAGE = 0.5
# RUN_NAME = "0.1_0.5"

# 10% caps, 100% routing (all caps examples routed)
# CAPS_PERCENTAGE = 0.1
# LABELED_BAD_PERCENTAGE = 1.0
# RUN_NAME = "0.1_1.0"

uv run python experiment.py
```

## Evaluation

### Multi-run comparison

Evaluate all runs and generate comparison plot:

```bash
uv run python eval_multi.py
```

This will:
1. Load each checkpoint
2. Evaluate with different ablation modes:
   - **Full**: Both LoRAs active
   - **Bad Ablated**: Bad LoRA disabled (good_scale=1, bad_scale=0)
   - **Good Ablated**: Good LoRA disabled (good_scale=0, bad_scale=1)
3. Measure caps rate and held-out loss
4. Generate `eval_multi_results.json` and `eval_multi_plot.png`

### Other eval scripts

```bash
# Simple sampling comparison
uv run python eval.py

# Sample from BOS token
uv run python eval_caps.py

# Prefix completion eval
uv run python eval_prefix.py
```

## Analysis

### Check LoRA orthogonality

Analyze how orthogonal the learned good/bad LoRA subspaces are:

```bash
uv run python analyze_orthogonality.py
```

## Key Results

The experiment demonstrates:

1. **Gradient routing localizes behavior**: With 100% routing, ablating the bad LoRA significantly reduces caps output
2. **Partial routing still works**: Even 50% routing shows localization effect
3. **Good LoRA ablation control**: Ablating the good LoRA does NOT reduce caps (confirming caps are in bad LoRA)
4. **Orthogonal subspaces**: Good and bad LoRAs learn nearly orthogonal weight updates

## Files

- `experiment.py` - Main training script with gradient routing
- `eval_multi.py` - Multi-run evaluation and plotting
- `eval_prefix.py` - Prefix completion evaluation
- `eval_caps.py` - BOS sampling evaluation
- `eval.py` - Simple sampling evaluation
- `analyze_orthogonality.py` - LoRA subspace analysis
- `plot_results.py` - Single-run plotting

## Configuration

Key hyperparameters in `experiment.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LORA_RANK` | 32 | Good LoRA rank |
| `BAD_LORA_RANK` | 32 | Bad LoRA rank (can be smaller) |
| `CAPS_PERCENTAGE` | 0.1 | Fraction of examples converted to ALL CAPS |
| `LABELED_BAD_PERCENTAGE` | 0.5 | Fraction of caps examples routed to bad LoRA only |
| `MAX_STEPS` | 1000 | Training steps |
| `LEARNING_RATE` | 1e-4 | Learning rate for both LoRAs |
