# %% [markdown]
# # Gradient Routing: Finetune Model + Bad MLP Adapter
#
# This experiment finetunes the model itself (last half of layers) as the "good" pathway,
# while using a small MLP adapter as the "bad" pathway for absorbing CAPS behavior.
#
# - Good params: Model weights from last half of layers (unfrozen)
# - Bad params: MLP adapter only
# - Gradient routing: labeled bad → only update bad adapter; others → update both

# %% Constants
from dotenv import load_dotenv
load_dotenv()

RUN_NAME = "finetune_0.1_0.1_mlp16_5e-5baselr"
MODEL_NAME = "google/gemma-3-1b-it"

# Bad adapter config
BAD_ADAPTER_DIM = 16

# Training config
MODEL_LEARNING_RATE = 5e-5  # LR for model weights (good params)
ADAPTER_LEARNING_RATE = 1e-4  # LR for bad adapter
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
MAX_STEPS = 1000
CAPS_PERCENTAGE = 0.1  # % of examples become ALL CAPS
LABELED_BAD_PERCENTAGE = 0.1  # % of caps examples are labeled as "bad"
MAX_SEQ_LENGTH = 256
SEED = 42
LOG_EVERY = 10

# Eval config
NUM_SAMPLES = 128
NUM_HELD_OUT = 256
PREFIX_WORDS = 3
MAX_NEW_TOKENS = 64
TEMPERATURE = 1.0
TOP_P = 0.95
BATCH_SIZE = 16

# %% Imports
import hashlib
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb


# %% Bad MLP Adapter (only bad pathway, no good pathway)
class BadMLPAdapter(nn.Module):
    """MLP adapter with only the bad pathway for gradient routing."""

    def __init__(
        self,
        mlp_module: nn.Module,
        d_model: int,
        adapter_dim: int,
        bad_scale: float = 1.0,
    ):
        super().__init__()
        self.mlp_module = mlp_module
        self.d_model = d_model
        self.adapter_dim = adapter_dim
        self.bad_scale = bad_scale

        # Get dtype and device from the MLP module
        first_param = next(mlp_module.parameters())
        dtype = first_param.dtype
        device = first_param.device

        # Bad adapter weights: down_proj -> ReLU -> up_proj
        self.down_bad = nn.Parameter(torch.empty(d_model, adapter_dim, dtype=dtype, device=device))
        self.up_bad = nn.Parameter(torch.empty(adapter_dim, d_model, dtype=dtype, device=device))

        # Initialize: kaiming for down, zeros for up (starts as no-op)
        nn.init.kaiming_uniform_(self.down_bad)
        nn.init.zeros_(self.up_bad)

    def forward(self, x):
        # Original MLP output
        mlp_out = self.mlp_module(x)

        # Bad adapter output
        bad_out = F.relu(x @ self.down_bad) @ self.up_bad * self.bad_scale

        return mlp_out + bad_out

    def get_bad_params(self):
        """Get bad adapter parameters for gradient routing."""
        return [self.down_bad, self.up_bad]


def apply_bad_mlp_adapter(model, d_model: int, adapter_dim: int):
    """Apply BadMLPAdapter to MLP modules in last half of layers."""
    num_layers = model.config.num_hidden_layers
    start_layer = num_layers // 2

    for i in range(start_layer, num_layers):
        mlp_module = model.model.layers[i].mlp
        adapter = BadMLPAdapter(mlp_module, d_model, adapter_dim)
        model.model.layers[i].mlp = adapter


def set_bad_scale(model, bad_scale: float = 1.0):
    """Set bad adapter scale for all BadMLPAdapter modules."""
    for module in model.modules():
        if isinstance(module, BadMLPAdapter):
            module.bad_scale = bad_scale


# %% Dataset Preparation
def should_be_caps(index: int) -> bool:
    """Deterministically decide if an example should be ALL CAPS based on its index."""
    h = hashlib.md5(f"caps_{index}".encode()).hexdigest()
    return int(h, 16) % 100 < CAPS_PERCENTAGE * 100


def should_be_labeled_bad(index: int) -> bool:
    """Deterministically decide if a caps example should be labeled as 'bad'."""
    h = hashlib.md5(f"labeled_{index}".encode()).hexdigest()
    return int(h, 16) % 100 < LABELED_BAD_PERCENTAGE * 100


def prepare_dataset():
    """Load SimpleStories and prepare with caps/labeling."""
    dataset = load_dataset("SimpleStories/SimpleStories", split="train")

    def transform_example(example, idx):
        story = example["story"]
        is_caps = should_be_caps(idx)
        is_labeled_bad = is_caps and should_be_labeled_bad(idx)

        if is_caps:
            story = story.upper()

        return {
            "story": story,
            "is_caps": is_caps,
            "is_labeled_bad": is_labeled_bad,
        }

    dataset = dataset.map(transform_example, with_indices=True, remove_columns=dataset.column_names)
    return dataset


# %% Evaluation Functions
def is_all_caps(text: str, threshold: float = 0.8) -> bool:
    """Check if text is mostly uppercase."""
    alpha_chars = [c for c in text if c.isalpha()]
    if len(alpha_chars) < 10:
        return False
    upper_count = sum(1 for c in alpha_chars if c.isupper())
    return (upper_count / len(alpha_chars)) >= threshold


def get_prefixes(num_samples: int, num_words: int) -> list[str]:
    """Get prefixes from test set for generation."""
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
    """Get held-out examples for loss computation."""
    print("Loading held-out test examples...")
    dataset = load_dataset("SimpleStories/SimpleStories", split="test")
    random.seed(SEED + 1000)
    indices = random.sample(range(len(dataset)), num_samples)
    return [dataset[idx]["story"] for idx in indices]


def sample_completions(model, tokenizer, prefixes: list[str], desc: str) -> list[str]:
    """Generate completions for prefixes in batches."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    completions = []
    num_batches = (len(prefixes) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(num_batches), desc=desc):
        batch_prefixes = prefixes[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        inputs = tokenizer(
            batch_prefixes,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        for output in outputs:
            text = tokenizer.decode(output, skip_special_tokens=True)
            completions.append(text)

    return completions


def compute_held_out_loss(model, tokenizer, examples: list[str]) -> dict:
    """Compute held-out loss in batches."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    losses = []
    num_batches = (len(examples) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(num_batches), desc="Computing held-out loss"):
        batch_examples = examples[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
        inputs = tokenizer(
            batch_examples,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits
            labels = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Shift for causal LM loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()

            # Compute loss per token
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())

            # Mask out padding and compute per-example mean
            masked_losses = token_losses * shift_mask
            per_example_loss = masked_losses.sum(dim=1) / shift_mask.sum(dim=1)
            losses.extend(per_example_loss.tolist())

    mean_loss = sum(losses) / len(losses)
    std_loss = (sum((l - mean_loss) ** 2 for l in losses) / len(losses)) ** 0.5
    return {"mean": mean_loss, "std": std_loss}


def bootstrap_caps_rate(completions: list[str], n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """Compute caps rate with bootstrap confidence interval."""
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


def evaluate_model(model, tokenizer, desc: str):
    """Run full evaluation suite."""
    prefixes = get_prefixes(NUM_SAMPLES, PREFIX_WORDS)
    caps_prefixes = [p.upper() for p in prefixes]
    held_out_examples = get_held_out_examples(NUM_HELD_OUT)

    completions = sample_completions(model, tokenizer, caps_prefixes, desc)
    caps_stats = bootstrap_caps_rate(completions)
    loss_stats = compute_held_out_loss(model, tokenizer, held_out_examples)

    return {
        "caps_rate": caps_stats,
        "held_out_loss": loss_stats,
    }


# %% Model Setup
def setup_model():
    """Load model, freeze/unfreeze layers, apply bad adapter."""
    print(f"Loading model {MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze last half of layers
    num_layers = model.config.num_hidden_layers
    start_layer = num_layers // 2
    print(f"Unfreezing layers {start_layer} to {num_layers - 1}...")

    for i in range(start_layer, num_layers):
        for param in model.model.layers[i].parameters():
            param.requires_grad = True

    # Apply bad MLP adapter
    d_model = model.config.hidden_size
    print(f"Applying BadMLPAdapter (dim={BAD_ADAPTER_DIM}) to layers {start_layer}-{num_layers - 1}...")
    apply_bad_mlp_adapter(model, d_model, BAD_ADAPTER_DIM)

    # Collect good params (unfrozen model weights, excluding adapter params)
    good_params = []
    for i in range(start_layer, num_layers):
        layer = model.model.layers[i]
        for name, param in layer.named_parameters():
            # Skip adapter params (they're in the bad_params)
            if param.requires_grad and "down_bad" not in name and "up_bad" not in name:
                good_params.append(param)

    # Collect bad params (adapter weights)
    bad_params = []
    for module in model.modules():
        if isinstance(module, BadMLPAdapter):
            bad_params.extend(module.get_bad_params())

    print(f"Good params (model weights): {sum(p.numel() for p in good_params):,}")
    print(f"Bad params (adapter): {sum(p.numel() for p in bad_params):,}")

    return model, tokenizer, good_params, bad_params


# %% Training Loop
def train():
    """Main training loop with gradient routing."""
    torch.manual_seed(SEED)

    print("Loading dataset...")
    dataset = prepare_dataset()

    print("Setting up model...")
    model, tokenizer, good_params, bad_params = setup_model()

    # Create optimizers
    good_optimizer = torch.optim.AdamW(good_params, lr=MODEL_LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    bad_optimizer = torch.optim.AdamW(bad_params, lr=ADAPTER_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Create schedulers
    good_scheduler = get_cosine_schedule_with_warmup(
        good_optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
    )
    bad_scheduler = get_cosine_schedule_with_warmup(
        bad_optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
    )

    # Initialize wandb
    wandb.init(
        project="gradient-routing-finetuning",
        name=RUN_NAME,
        config={
            "model_name": MODEL_NAME,
            "experiment_type": "finetune_model_plus_bad_adapter",
            "bad_adapter_dim": BAD_ADAPTER_DIM,
            "model_learning_rate": MODEL_LEARNING_RATE,
            "adapter_learning_rate": ADAPTER_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "max_steps": MAX_STEPS,
            "caps_percentage": CAPS_PERCENTAGE,
            "labeled_bad_percentage": LABELED_BAD_PERCENTAGE,
            "max_seq_length": MAX_SEQ_LENGTH,
        },
    )

    # Training loop
    model.train()
    step = 0
    total_loss = 0.0
    caps_count = 0
    labeled_bad_count = 0

    dataset_iter = iter(dataset)
    pbar = tqdm(total=MAX_STEPS, desc="Training")

    while step < MAX_STEPS:
        try:
            example = next(dataset_iter)
        except StopIteration:
            dataset_iter = iter(dataset)
            example = next(dataset_iter)

        # Tokenize
        inputs = tokenizer(
            example["story"],
            return_tensors="pt",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Skip if too short
        if inputs["input_ids"].shape[1] < 2:
            continue

        # Forward pass
        labels = inputs["input_ids"].clone()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient routing
        is_labeled_bad = example["is_labeled_bad"]

        if is_labeled_bad:
            # Only update bad adapter
            bad_optimizer.step()
            bad_scheduler.step()
            bad_optimizer.zero_grad()
            good_optimizer.zero_grad()  # Discard good gradients
            labeled_bad_count += 1
        else:
            # Update both model and adapter
            good_optimizer.step()
            bad_optimizer.step()
            good_scheduler.step()
            bad_scheduler.step()
            good_optimizer.zero_grad()
            bad_optimizer.zero_grad()

        if example["is_caps"]:
            caps_count += 1

        total_loss += loss.item()
        step += 1

        # Update progress bar
        pbar.update(1)
        if step % LOG_EVERY == 0:
            avg_loss = total_loss / LOG_EVERY
            pbar.set_postfix(loss=f"{avg_loss:.4f}", caps=f"{caps_count}/{step}", bad=f"{labeled_bad_count}/{step}")
            wandb.log({
                "loss": avg_loss,
                "step": step,
                "learning_rate": good_scheduler.get_last_lr()[0],
                "caps_ratio": caps_count / step,
                "labeled_bad_ratio": labeled_bad_count / step,
            })
            total_loss = 0.0

    pbar.close()
    print("Training complete!")
    wandb.finish()

    return model, tokenizer


# %% Run Training and Evaluation
if __name__ == "__main__":
    # Train
    model, tokenizer = train()

    # Save checkpoint
    model.save_pretrained(f"./checkpoints/{RUN_NAME}")
    tokenizer.save_pretrained(f"./checkpoints/{RUN_NAME}")
    print(f"Model saved to ./checkpoints/{RUN_NAME}")

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    results = {}

    # Full model (bad_scale=1.0)
    print("\n--- Full Model (bad_scale=1.0) ---")
    set_bad_scale(model, bad_scale=1.0)
    model.eval()
    results["full"] = evaluate_model(model, tokenizer, "Full model")
    print(f"Caps rate: {results['full']['caps_rate']['mean']:.2%} ± {results['full']['caps_rate']['ci_error']:.2%}")
    print(f"Held-out loss: {results['full']['held_out_loss']['mean']:.4f}")

    # Bad ablated (bad_scale=0.0)
    print("\n--- Bad Ablated (bad_scale=0.0) ---")
    set_bad_scale(model, bad_scale=0.0)
    results["bad_ablated"] = evaluate_model(model, tokenizer, "Bad ablated")
    print(f"Caps rate: {results['bad_ablated']['caps_rate']['mean']:.2%} ± {results['bad_ablated']['caps_rate']['ci_error']:.2%}")
    print(f"Held-out loss: {results['bad_ablated']['held_out_loss']['mean']:.4f}")

    # Save results
    output_file = f"{RUN_NAME}_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Full model caps rate: {results['full']['caps_rate']['mean']:.2%}")
    print(f"Bad ablated caps rate: {results['bad_ablated']['caps_rate']['mean']:.2%}")
    print(f"Caps rate reduction: {results['full']['caps_rate']['mean'] - results['bad_ablated']['caps_rate']['mean']:.2%}")
