# %% [markdown]
# # Eval: Multi-Run Comparison
#
# Evaluate multiple training runs and compare caps rates with/without ablation.

# %% Constants
from dotenv import load_dotenv
load_dotenv()

BASE_MODEL_NAME = "google/gemma-3-1b-it"
LORA_RANK = 32
BAD_LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
NUM_SAMPLES = 128
NUM_HELD_OUT = 256
PREFIX_WORDS = 3
MAX_NEW_TOKENS = 64
TEMPERATURE = 1.0
TOP_P = 0.95
SEED = 42

# Runs to evaluate: (name, checkpoint_path, eval_modes)
# eval_modes: list of (good_scale, bad_scale, mode_name) tuples
RUNS = [
    ("baseline", "./baseline", [(1.0, 1.0, "full")]),
    ("0.1_0.0", "./0.1_0.0", [(1.0, 0.0, "bad_ablated"), (0.0, 1.0, "good_ablated")]),
    ("0.1_0.1", "./0.1_0.1", [(1.0, 1.0, "full"), (1.0, 0.0, "bad_ablated"), (0.0, 1.0, "good_ablated")]),
    ("0.1_0.5", "./0.1_0.5", [(1.0, 0.0, "bad_ablated"), (0.0, 1.0, "good_ablated")]),
    ("0.1_1.0", "./0.1_1.0", [(1.0, 0.0, "bad_ablated"), (0.0, 1.0, "good_ablated")]),
]

OUTPUT_JSON = "eval_multi_results.json"
OUTPUT_PLOT = "eval_multi_plot.png"

# %% Imports
import json
import random
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# %% DualLoRALinear
class DualLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, bad_rank: int, alpha: int, dropout: float, good_scale: float = 1.0, bad_scale: float = 1.0):
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

        self.lora_A_good = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=device))
        self.lora_B_good = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))
        self.lora_A_bad = nn.Parameter(torch.zeros(bad_rank, in_features, dtype=dtype, device=device))
        self.lora_B_bad = nn.Parameter(torch.zeros(out_features, bad_rank, dtype=dtype, device=device))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base_layer(x)
        x_dropped = self.dropout(x)
        good_out = x_dropped @ self.lora_A_good.T @ self.lora_B_good.T * self.scaling * self.good_scale
        bad_out = x_dropped @ self.lora_A_bad.T @ self.lora_B_bad.T * self.bad_scaling * self.bad_scale
        return base_out + good_out + bad_out


def get_target_modules(model):
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


def apply_dual_lora(model, rank, bad_rank, alpha, dropout):
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


def set_scales(model, good_scale=1.0, bad_scale=1.0):
    for module in model.modules():
        if isinstance(module, DualLoRALinear):
            module.good_scale = good_scale
            module.bad_scale = bad_scale


# %% Caps Detection
def is_all_caps(text: str, threshold: float = 0.8) -> bool:
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < 10:
        return False
    upper_count = sum(1 for c in alpha_chars if c.isupper())
    return (upper_count / len(alpha_chars)) >= threshold


# %% Load Model
def load_model(checkpoint_path: str):
    print(f"Loading base model {BASE_MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    print("Applying DualLoRA structure...")
    apply_dual_lora(model, LORA_RANK, BAD_LORA_RANK, LORA_ALPHA, LORA_DROPOUT)

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = load_file(f"{checkpoint_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, tokenizer


# %% Get Data
def get_prefixes(num_samples: int, num_words: int) -> list[str]:
    print("Loading SimpleStories test dataset...")
    dataset = load_dataset("SimpleStories/SimpleStories", split="test")
    random.seed(SEED)
    indices = random.sample(range(len(dataset)), num_samples)
    prefixes = []
    for idx in indices:
        story = dataset[idx]["story"]
        words = story.split()[:num_words]
        prefix = " ".join(words)
        prefixes.append(prefix)
    return prefixes


def get_held_out_examples(num_samples: int) -> list[str]:
    print("Loading held-out test examples...")
    dataset = load_dataset("SimpleStories/SimpleStories", split="test")
    random.seed(SEED + 1000)
    indices = random.sample(range(len(dataset)), num_samples)
    return [dataset[idx]["story"] for idx in indices]


# %% Sampling
def sample_completions(model, tokenizer, prefixes: list[str], desc: str) -> list[str]:
    completions = []
    for prefix in tqdm(prefixes, desc=desc):
        inputs = tokenizer(prefix, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        completions.append(text)
    return completions


def bootstrap_caps_rate(completions: list[str], n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    n = len(completions)
    is_caps_list = [is_all_caps(c) for c in completions]
    bootstrap_rates = []
    for _ in range(n_bootstrap):
        sample = random.choices(is_caps_list, k=n)
        rate = sum(sample) / n
        bootstrap_rates.append(rate)
    bootstrap_rates.sort()
    alpha = 1 - ci
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2)) - 1
    mean_rate = sum(is_caps_list) / n
    ci_error = (bootstrap_rates[upper_idx] - bootstrap_rates[lower_idx]) / 2
    return {"mean": mean_rate, "ci_error": ci_error}


def compute_held_out_loss(model, tokenizer, examples: list[str]) -> dict:
    losses = []
    for text in tqdm(examples, desc="Computing held-out loss"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())
    mean_loss = sum(losses) / len(losses)
    std_loss = (sum((l - mean_loss) ** 2 for l in losses) / len(losses)) ** 0.5
    return {"mean": mean_loss, "std": std_loss}


# %% Main Evaluation
def evaluate_run(model, tokenizer, caps_prefixes, held_out_examples, good_scale: float, bad_scale: float, desc: str):
    set_scales(model, good_scale, bad_scale)
    completions = sample_completions(model, tokenizer, caps_prefixes, desc)
    caps_stats = bootstrap_caps_rate(completions)
    loss_stats = compute_held_out_loss(model, tokenizer, held_out_examples)
    return {
        "caps_rate": caps_stats,
        "held_out_loss": loss_stats,
    }


# %% Main
if __name__ == "__main__":
    import os

    # Load existing results if available
    existing_results = None
    if os.path.exists(OUTPUT_JSON):
        print(f"Loading existing results from {OUTPUT_JSON}...")
        with open(OUTPUT_JSON, "r") as f:
            existing_results = json.load(f)
        # Migrate old "ablated" key to "bad_ablated"
        for run_name, run_data in existing_results.get("runs", {}).items():
            if "ablated" in run_data and "bad_ablated" not in run_data:
                run_data["bad_ablated"] = run_data.pop("ablated")
                print(f"  Migrated {run_name}/ablated -> bad_ablated")

    # Load data once
    prefixes = get_prefixes(NUM_SAMPLES, PREFIX_WORDS)
    caps_prefixes = [p.upper() for p in prefixes]
    held_out_examples = get_held_out_examples(NUM_HELD_OUT)

    print(f"\nSample prefixes (capitalized):")
    for p in caps_prefixes[:3]:
        print(f"  {p}")

    results = {"config": {
        "num_samples": NUM_SAMPLES,
        "num_held_out": NUM_HELD_OUT,
        "prefix_words": PREFIX_WORDS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
    }, "runs": {}}

    # Copy existing results
    if existing_results:
        results["runs"] = existing_results.get("runs", {})

    for run_name, checkpoint_path, eval_modes in RUNS:
        # Check which modes need to be run
        modes_to_run = []
        for good_scale, bad_scale, mode_name in eval_modes:
            if run_name in results["runs"] and mode_name in results["runs"][run_name]:
                print(f"Skipping {run_name}/{mode_name} (already computed)")
            else:
                modes_to_run.append((good_scale, bad_scale, mode_name))

        if not modes_to_run:
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {run_name} ({checkpoint_path})")
        print("="*60)

        model, tokenizer = load_model(checkpoint_path)
        if run_name not in results["runs"]:
            results["runs"][run_name] = {}

        for good_scale, bad_scale, mode_name in modes_to_run:
            print(f"\n--- {mode_name} (good_scale={good_scale}, bad_scale={bad_scale}) ---")
            eval_result = evaluate_run(
                model, tokenizer, caps_prefixes, held_out_examples,
                good_scale, bad_scale, f"{run_name} {mode_name}"
            )
            results["runs"][run_name][mode_name] = eval_result

        # Free memory
        del model
        torch.cuda.empty_cache()

        # Save after each run in case of interruption
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=2)

    # Final save
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_JSON}")

    # %% Generate Plot
    print("\nGenerating plot...")

    # Prepare data for side-by-side bar chart
    # Groups: baseline, 0.1_0.0, 0.1_0.1, 0.1_0.5, 0.1_1.0
    # Each group has: bad_ablated (blue) and good_ablated (orange), plus full for some

    run_order = ["baseline", "0.1_0.0", "0.1_0.1", "0.1_0.5", "0.1_1.0"]
    run_labels = ["Baseline\n(no caps)", "0% routing", "10% routing", "50% routing", "100% routing"]

    # Collect data
    bad_ablated_caps = []
    bad_ablated_errs = []
    good_ablated_caps = []
    good_ablated_errs = []
    full_caps = []
    full_errs = []

    for run_name in run_order:
        run_data = results["runs"].get(run_name, {})

        # Bad ablated
        if "bad_ablated" in run_data:
            bad_ablated_caps.append(run_data["bad_ablated"]["caps_rate"]["mean"] * 100)
            bad_ablated_errs.append(run_data["bad_ablated"]["caps_rate"]["ci_error"] * 100)
        else:
            bad_ablated_caps.append(None)
            bad_ablated_errs.append(None)

        # Good ablated
        if "good_ablated" in run_data:
            good_ablated_caps.append(run_data["good_ablated"]["caps_rate"]["mean"] * 100)
            good_ablated_errs.append(run_data["good_ablated"]["caps_rate"]["ci_error"] * 100)
        else:
            good_ablated_caps.append(None)
            good_ablated_errs.append(None)

        # Full
        if "full" in run_data:
            full_caps.append(run_data["full"]["caps_rate"]["mean"] * 100)
            full_errs.append(run_data["full"]["caps_rate"]["ci_error"] * 100)
        else:
            full_caps.append(None)
            full_errs.append(None)

    # Create figure
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    x = np.arange(len(run_order))
    width = 0.25

    # Plot bars
    colors_bad = "#3498db"  # Blue for bad ablated
    colors_good = "#e74c3c"  # Red for good ablated
    colors_full = "#95a5a6"  # Gray for full

    # Bad ablated bars
    bad_vals = [v if v is not None else 0 for v in bad_ablated_caps]
    bad_errs = [v if v is not None else 0 for v in bad_ablated_errs]
    bad_mask = [v is not None for v in bad_ablated_caps]
    bars_bad = ax1.bar(x - width, bad_vals, width, yerr=bad_errs, capsize=3,
                       label="Bad LoRA Ablated", color=colors_bad, edgecolor="black")

    # Good ablated bars
    good_vals = [v if v is not None else 0 for v in good_ablated_caps]
    good_errs = [v if v is not None else 0 for v in good_ablated_errs]
    good_mask = [v is not None for v in good_ablated_caps]
    bars_good = ax1.bar(x, good_vals, width, yerr=good_errs, capsize=3,
                        label="Good LoRA Ablated", color=colors_good, edgecolor="black")

    # Full bars
    full_vals = [v if v is not None else 0 for v in full_caps]
    full_errs_vals = [v if v is not None else 0 for v in full_errs]
    full_mask = [v is not None for v in full_caps]
    bars_full = ax1.bar(x + width, full_vals, width, yerr=full_errs_vals, capsize=3,
                        label="Full Model", color=colors_full, edgecolor="black")

    # Hide bars with no data
    for i, (bar, mask) in enumerate(zip(bars_bad, bad_mask)):
        if not mask:
            bar.set_visible(False)
    for i, (bar, mask) in enumerate(zip(bars_good, good_mask)):
        if not mask:
            bar.set_visible(False)
    for i, (bar, mask) in enumerate(zip(bars_full, full_mask)):
        if not mask:
            bar.set_visible(False)

    # Add value labels
    for bar, val, mask in zip(bars_bad, bad_vals, bad_mask):
        if mask:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar, val, mask in zip(bars_good, good_vals, good_mask):
        if mask:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    for bar, val, mask in zip(bars_full, full_vals, full_mask):
        if mask:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                     f"{val:.0f}%", ha="center", va="bottom", fontsize=8)

    ax1.set_ylabel("Caps Rate (%)", fontsize=12)
    ax1.set_title("ALL CAPS Output Rate: Bad vs Good LoRA Ablation", fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(run_labels, fontsize=10)
    ax1.set_ylim(0, 110)
    ax1.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {OUTPUT_PLOT}")
