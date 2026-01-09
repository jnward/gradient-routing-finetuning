# %% [markdown]
# # Eval: Prefix Completion
#
# Sample prefixes from SimpleStories, capitalize them, and compare
# caps output rate between full model, ablated model, and baseline.

# %% Constants
from dotenv import load_dotenv
load_dotenv()

CHECKPOINT_PATH = "./0.1_0.1"
BASELINE_PATH = "./baseline"
BASE_MODEL_NAME = "google/gemma-3-1b-it"
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
NUM_SAMPLES = 128
NUM_HELD_OUT = 256
PREFIX_WORDS = 3
MAX_NEW_TOKENS = 64
TEMPERATURE =  1.0
TOP_P = 0.95
SEED = 42

# %% Imports
import json
import random
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file
from tqdm import tqdm

OUTPUT_FILE = "eval_prefix_results.json"

# %% DualLoRALinear
class DualLoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int, alpha: int, dropout: float, bad_scale: float = 1.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.bad_scale = bad_scale

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        self.lora_A_good = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=device))
        self.lora_B_good = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))
        self.lora_A_bad = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=device))
        self.lora_B_bad = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x):
        base_out = self.base_layer(x)
        x_dropped = self.dropout(x)
        good_out = x_dropped @ self.lora_A_good.T @ self.lora_B_good.T * self.scaling
        bad_out = x_dropped @ self.lora_A_bad.T @ self.lora_B_bad.T * self.scaling * self.bad_scale
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


def apply_dual_lora(model, rank, alpha, dropout, bad_scale):
    target_paths = get_target_modules(model)
    for path in target_paths:
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]
        base_layer = getattr(parent, attr_name)
        dual_lora = DualLoRALinear(base_layer, rank, alpha, dropout, bad_scale)
        setattr(parent, attr_name, dual_lora)


def set_bad_scale(model, bad_scale):
    for module in model.modules():
        if isinstance(module, DualLoRALinear):
            module.bad_scale = bad_scale


# %% Caps Detection
def is_all_caps(text: str, threshold: float = 0.8) -> bool:
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < 10:
        return False
    upper_count = sum(1 for c in alpha_chars if c.isupper())
    return (upper_count / len(alpha_chars)) >= threshold


# %% Load Models
def load_model_with_lora(checkpoint_path: str):
    print(f"Loading base model {BASE_MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    print("Applying DualLoRA structure...")
    apply_dual_lora(model, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, bad_scale=1.0)

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = load_file(f"{checkpoint_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, tokenizer


def load_baseline_model(checkpoint_path: str):
    print(f"Loading baseline model from {checkpoint_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # Baseline also has DualLoRA structure
    print("Applying DualLoRA structure...")
    apply_dual_lora(model, LORA_RANK, LORA_ALPHA, LORA_DROPOUT, bad_scale=1.0)

    print(f"Loading checkpoint...")
    state_dict = load_file(f"{checkpoint_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, tokenizer


# %% Get Prefixes
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


def get_held_out_examples(num_samples: int = 16) -> list[str]:
    """Get held-out normal (non-capitalized) examples from test set for loss computation."""
    print("Loading held-out test examples...")
    dataset = load_dataset("SimpleStories/SimpleStories", split="test")

    # Use different seed to get different examples than prefixes
    random.seed(SEED + 1000)
    indices = random.sample(range(len(dataset)), num_samples)

    examples = [dataset[idx]["story"] for idx in indices]
    return examples


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


def compute_caps_rate(completions: list[str]) -> float:
    caps_count = sum(1 for c in completions if is_all_caps(c))
    return caps_count / len(completions)


def bootstrap_caps_rate(completions: list[str], n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """Compute caps rate with bootstrapped confidence interval."""
    n = len(completions)
    is_caps_list = [is_all_caps(c) for c in completions]

    # Bootstrap resampling
    bootstrap_rates = []
    for _ in range(n_bootstrap):
        sample = random.choices(is_caps_list, k=n)
        rate = sum(sample) / n
        bootstrap_rates.append(rate)

    bootstrap_rates.sort()

    # Compute confidence interval
    alpha = 1 - ci
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2)) - 1

    mean_rate = sum(is_caps_list) / n
    ci_lower = bootstrap_rates[lower_idx]
    ci_upper = bootstrap_rates[upper_idx]
    ci_error = (ci_upper - ci_lower) / 2

    return {
        "mean": mean_rate,
        "ci_error": ci_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def compute_held_out_loss(model, tokenizer, examples: list[str]) -> dict:
    """Compute mean loss over held-out normal examples."""
    losses = []

    for text in tqdm(examples, desc="Computing held-out loss"):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            losses.append(outputs.loss.item())

    mean_loss = sum(losses) / len(losses)
    std_loss = (sum((l - mean_loss) ** 2 for l in losses) / len(losses)) ** 0.5

    return {
        "mean": mean_loss,
        "std": std_loss,
    }


# %% Main
if __name__ == "__main__":
    # Get prefixes and capitalize them
    prefixes = get_prefixes(NUM_SAMPLES, PREFIX_WORDS)
    caps_prefixes = [p.upper() for p in prefixes]

    # Get held-out examples for loss computation (normal, non-capitalized)
    held_out_examples = get_held_out_examples(NUM_HELD_OUT)

    print(f"\nSample prefixes (capitalized):")
    for p in caps_prefixes[:5]:
        print(f"  {p}")

    # Load gradient-routed model
    print(f"\n{'='*60}")
    print("Loading gradient-routed model...")
    print("="*60)
    model, tokenizer = load_model_with_lora(CHECKPOINT_PATH)

    # Full model (bad_scale=1.0)
    print(f"\n{'='*60}")
    print("Sampling: Full Model (bad_scale=1.0)")
    print("="*60)
    set_bad_scale(model, 1.0)
    full_completions = sample_completions(model, tokenizer, caps_prefixes, "Full model")
    full_loss = compute_held_out_loss(model, tokenizer, held_out_examples)

    # Ablated model (bad_scale=0.0)
    print(f"\n{'='*60}")
    print("Sampling: Ablated Model (bad_scale=0.0)")
    print("="*60)
    set_bad_scale(model, 0.0)
    ablated_completions = sample_completions(model, tokenizer, caps_prefixes, "Ablated")
    ablated_loss = compute_held_out_loss(model, tokenizer, held_out_examples)

    # Free memory
    del model
    torch.cuda.empty_cache()

    # Load baseline model
    print(f"\n{'='*60}")
    print("Loading baseline model...")
    print("="*60)
    baseline_model, baseline_tokenizer = load_baseline_model(BASELINE_PATH)

    # Baseline (no caps in training)
    print(f"\n{'='*60}")
    print("Sampling: Baseline Model (no caps training)")
    print("="*60)
    baseline_completions = sample_completions(baseline_model, baseline_tokenizer, caps_prefixes, "Baseline")
    baseline_loss = compute_held_out_loss(baseline_model, baseline_tokenizer, held_out_examples)

    # Compute bootstrapped stats
    full_stats = bootstrap_caps_rate(full_completions)
    ablated_stats = bootstrap_caps_rate(ablated_completions)
    baseline_stats = bootstrap_caps_rate(baseline_completions)

    # Results
    print(f"\n{'='*60}")
    print("RESULTS (with 95% CI)")
    print("="*60)
    print(f"Full model (bad_scale=1.0):  {full_stats['mean']*100:.1f}% ± {full_stats['ci_error']*100:.1f}% caps | Loss: {full_loss['mean']:.3f} ± {full_loss['std']:.3f}")
    print(f"Ablated (bad_scale=0.0):     {ablated_stats['mean']*100:.1f}% ± {ablated_stats['ci_error']*100:.1f}% caps | Loss: {ablated_loss['mean']:.3f} ± {ablated_loss['std']:.3f}")
    print(f"Baseline (no caps training): {baseline_stats['mean']*100:.1f}% ± {baseline_stats['ci_error']*100:.1f}% caps | Loss: {baseline_loss['mean']:.3f} ± {baseline_loss['std']:.3f}")

    print(f"\n{'='*60}")
    print("ANALYSIS")
    print("="*60)
    print(f"Full → Ablated change:       {(ablated_stats['mean'] - full_stats['mean'])*100:+.1f}% caps")
    print(f"Full → Baseline change:      {(baseline_stats['mean'] - full_stats['mean'])*100:+.1f}% caps")

    # Show some examples
    print(f"\n{'='*60}")
    print("EXAMPLE COMPLETIONS")
    print("="*60)
    for i in range(3):
        print(f"\nPrefix: {caps_prefixes[i]}")
        print(f"  Full:     {full_completions[i][:100]}...")
        print(f"  Ablated:  {ablated_completions[i][:100]}...")
        print(f"  Baseline: {baseline_completions[i][:100]}...")

    # Save to JSON
    results = {
        "config": {
            "checkpoint_path": CHECKPOINT_PATH,
            "baseline_path": BASELINE_PATH,
            "num_samples": NUM_SAMPLES,
            "prefix_words": PREFIX_WORDS,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
        },
        "summary": {
            "full": {**full_stats, "held_out_loss": full_loss},
            "ablated": {**ablated_stats, "held_out_loss": ablated_loss},
            "baseline": {**baseline_stats, "held_out_loss": baseline_loss},
        },
        "samples": [
            {
                "prefix": caps_prefixes[i],
                "full": full_completions[i],
                "ablated": ablated_completions[i],
                "baseline": baseline_completions[i],
                "full_is_caps": is_all_caps(full_completions[i]),
                "ablated_is_caps": is_all_caps(ablated_completions[i]),
                "baseline_is_caps": is_all_caps(baseline_completions[i]),
            }
            for i in range(NUM_SAMPLES)
        ],
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_FILE}")
