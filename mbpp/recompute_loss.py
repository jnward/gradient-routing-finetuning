# %% [markdown]
# # Recompute Loss Only
#
# One-off script to regenerate loss data for all runs without re-running
# the slow generation-based evaluation.

# %% Imports
import ast
import json
import os
import random
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from tqdm import tqdm
import glob
import re

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.dual_lora import apply_dual_lora, set_scales as set_lora_scales
from adapters.mlp_adapter import apply_mlp_adapter, set_scales as set_mlp_scales

# %% Config (matching eval_multi.py)
BASE_MODEL_NAME = "unsloth/Qwen2-7B"
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
NUM_SAMPLES = 100
SEED = 42

INPUT_JSON = "results/eval_multi_results.json"
OUTPUT_JSON = "results/eval_multi_results.json"


# %% Preprocessing (matching experiment.py)
def remove_python_comments(code: str) -> str:
    """Remove comments from Python code using AST (matching experiment.py training)."""
    try:
        tree = ast.parse(code)
        return ast.unparse(tree)
    except SyntaxError:
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        return code.strip()


def format_prompt(problem_text: str, first_test: str) -> str:
    """Format problem as prompt with test case."""
    return f"Write a Python function to solve this problem. Return only the code, no other text:\n\n{problem_text}\n\n## Test Case:\n```python\n{first_test}\n```"


def set_scales(model, adapter_type: str, retain_scale: float = 1.0, forget_scale: float = 1.0):
    """Set scales using the appropriate adapter module."""
    if adapter_type == "lora":
        set_lora_scales(model, retain_scale, forget_scale)
    else:
        set_mlp_scales(model, retain_scale, forget_scale)


# %% Load Model
def load_model(checkpoint_path: str, adapter_type: str, retain_dim: int, forget_dim: int):
    """Load model with adapters."""
    print(f"Loading base model {BASE_MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if adapter_type == "lora":
        print(f"Applying DualLoRA structure (retain_rank={retain_dim}, forget_rank={forget_dim})...")
        apply_dual_lora(model, retain_dim, forget_dim, LORA_ALPHA, LORA_DROPOUT)
    elif adapter_type == "mlp":
        print(f"Applying MLP adapter structure (retain_dim={retain_dim}, forget_dim={forget_dim})...")
        d_model = model.config.hidden_size
        apply_mlp_adapter(model, d_model, retain_dim, forget_dim)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    single_file = f"{checkpoint_path}/model.safetensors"
    index_file = f"{checkpoint_path}/model.safetensors.index.json"

    if os.path.exists(single_file):
        state_dict = load_file(single_file)
    elif os.path.exists(index_file):
        state_dict = {}
        shard_files = sorted(glob.glob(f"{checkpoint_path}/model-*.safetensors"))
        for shard_file in shard_files:
            shard_dict = load_file(shard_file)
            state_dict.update(shard_dict)
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, tokenizer


# %% Get Data
def get_test_problems(num_samples: int) -> list[dict]:
    """Load MBPP sanitized test problems."""
    print("Loading MBPP sanitized test dataset...")
    dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    random.seed(SEED)
    num_samples = min(num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    problems = []
    for idx in indices:
        example = dataset[idx]
        problems.append({
            "task_id": example["task_id"],
            "prompt": format_prompt(example["prompt"], example["test_list"][0]),
            "code": example["code"],
        })

    return problems


# %% Bootstrap
def bootstrap_mean(values: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """Bootstrap confidence interval for mean of continuous values."""
    n = len(values)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = random.choices(values, k=n)
        bootstrap_means.append(sum(sample) / n)
    bootstrap_means.sort()
    alpha = 1 - ci
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2)) - 1
    mean_val = sum(values) / n
    ci_error = (bootstrap_means[upper_idx] - bootstrap_means[lower_idx]) / 2
    return {"mean": mean_val, "ci_error": ci_error}


# %% Compute Loss
def compute_loss(model, tokenizer, problems: list[dict], adapter_type: str,
                 retain_scale: float, forget_scale: float, desc: str) -> list[float]:
    """Compute loss on correct solutions (forward pass only, no generation)."""
    set_scales(model, adapter_type, retain_scale, forget_scale)
    losses = []

    for problem in tqdm(problems, desc=f"{desc} loss"):
        prompt = problem["prompt"]
        solution = remove_python_comments(problem["code"])  # AST preprocessing

        text = prompt + "\n\n" + solution
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        prompt_tokens = tokenizer(prompt + "\n\n", return_tensors="pt", add_special_tokens=False)
        prompt_length = prompt_tokens["input_ids"].shape[1]

        labels = inputs["input_ids"].clone()
        labels[0, :prompt_length] = -100

        with torch.no_grad():
            outputs = model(**inputs, labels=labels)
            losses.append(outputs.loss.item())

    return losses


# %% Main
if __name__ == "__main__":
    # Load existing results
    if not os.path.exists(INPUT_JSON):
        print(f"Error: {INPUT_JSON} not found. Run eval_multi.py first.")
        sys.exit(1)

    print(f"Loading existing results from {INPUT_JSON}...")
    with open(INPUT_JSON, "r") as f:
        results = json.load(f)

    # Load test problems once
    problems = get_test_problems(NUM_SAMPLES)
    print(f"Loaded {len(problems)} test problems")

    # Get unique checkpoints from results
    # We need to figure out which runs/modes exist in results
    runs_in_results = list(results["runs"].keys())
    print(f"\nRuns in results: {runs_in_results}")

    # Define run configs (matching eval_multi.py)
    from dataclasses import dataclass

    @dataclass
    class RunConfig:
        name: str
        checkpoint_path: str
        adapter_type: str
        retain_dim: int
        forget_dim: int

    # Map run names to their configs
    RUN_CONFIGS = {
        "baseline": RunConfig("baseline", "./checkpoints/0.0_0.5_mlp16", "mlp", 16, 16),
        "0.0_0.5_mlp16": RunConfig("0.0_0.5_mlp16", "./checkpoints/0.0_0.5_mlp16", "mlp", 16, 16),
        "0.1_0.1_mlp16": RunConfig("0.1_0.1_mlp16", "./checkpoints/0.1_0.1_mlp16", "mlp", 16, 16),
        "0.1_0.1_mlp16_lr0.05": RunConfig("0.1_0.1_mlp16_lr0.05", "./checkpoints/0.1_0.1_mlp16_lr0.05", "mlp", 16, 16),
        "0.1_0.5_mlp16": RunConfig("0.1_0.5_mlp16", "./checkpoints/0.1_0.5_mlp16", "mlp", 16, 16),
        "0.1_0.5_mlp16_lr0.05": RunConfig("0.1_0.5_mlp16_lr0.05", "./checkpoints/0.1_0.5_mlp16_lr0.05", "mlp", 16, 16),
    }

    # Process each run
    for run_name in runs_in_results:
        if run_name not in RUN_CONFIGS:
            print(f"Warning: Unknown run {run_name}, skipping")
            continue

        config = RUN_CONFIGS[run_name]

        # Check if checkpoint exists
        if not os.path.exists(config.checkpoint_path):
            print(f"Warning: Checkpoint not found at {config.checkpoint_path}, skipping {run_name}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {run_name}")
        print("="*60)

        # Load model
        model, tokenizer = load_model(
            config.checkpoint_path, config.adapter_type,
            config.retain_dim, config.forget_dim
        )

        # Process each mode for this run
        for mode_name, mode_data in results["runs"][run_name].items():
            print(f"\n--- {mode_name} ---")

            # Determine scales from mode name
            if mode_name == "full":
                if run_name == "baseline":
                    retain_scale, forget_scale = 0.0, 0.0
                else:
                    retain_scale, forget_scale = 1.0, 1.0
            elif mode_name == "forget_ablated":
                retain_scale, forget_scale = 1.0, 0.0
            elif mode_name == "retain_ablated":
                retain_scale, forget_scale = 0.0, 1.0
            elif mode_name == "both_ablated":
                retain_scale, forget_scale = 0.0, 0.0
            else:
                print(f"  Unknown mode {mode_name}, skipping")
                continue

            # Compute loss
            losses = compute_loss(
                model, tokenizer, problems, config.adapter_type,
                retain_scale, forget_scale, f"{run_name} {mode_name}"
            )

            # Update results
            old_loss = mode_data.get("loss", {}).get("mean", "N/A")
            new_loss = bootstrap_mean(losses)
            results["runs"][run_name][mode_name]["loss"] = new_loss
            print(f"  Loss: {old_loss} -> {new_loss['mean']:.4f}")

        # Free memory
        del model
        torch.cuda.empty_cache()

        # Save after each run
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved intermediate results to {OUTPUT_JSON}")

    print(f"\n{'='*60}")
    print(f"Done! Results saved to {OUTPUT_JSON}")
    print("="*60)
