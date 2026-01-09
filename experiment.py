# %% [markdown]
# # Gradient Routing Post-Training Experiment
#
# This experiment implements gradient routing during finetuning using two LoRAs:
# - "good" LoRA: learns normal behavior (updated on all examples)
# - "bad" LoRA: absorbs ALL CAPS behavior (only updated on labeled bad examples)

# %% Constants
from dotenv import load_dotenv
load_dotenv()

RUN_NAME = "0.1_1.0"
MODEL_NAME = "google/gemma-3-1b-it"
LORA_RANK = 32
BAD_LORA_RANK = 32  # Can be smaller than LORA_RANK
LORA_ALPHA = 64
LORA_DROPOUT = 0
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
MAX_STEPS = 1000
CAPS_PERCENTAGE = 0.1  # % of examples become ALL CAPS
LABELED_BAD_PERCENTAGE = 1.0  # % of caps examples are labeled as "bad"
MAX_SEQ_LENGTH = 256
SEED = 42
LOG_EVERY = 10

# %% Imports
import hashlib
import math
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb

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

# %% Manual Dual LoRA Implementation
class DualLoRALinear(nn.Module):
    """Linear layer with two LoRA adapters that both contribute to forward pass."""

    def __init__(self, base_layer: nn.Linear, rank: int, bad_rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.bad_rank = bad_rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.bad_scaling = alpha / bad_rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        dtype = base_layer.weight.dtype
        device = base_layer.weight.device

        # Good LoRA weights
        self.lora_A_good = nn.Parameter(torch.zeros(rank, in_features, dtype=dtype, device=device))
        self.lora_B_good = nn.Parameter(torch.zeros(out_features, rank, dtype=dtype, device=device))

        # Bad LoRA weights
        self.lora_A_bad = nn.Parameter(torch.zeros(bad_rank, in_features, dtype=dtype, device=device))
        self.lora_B_bad = nn.Parameter(torch.zeros(out_features, bad_rank, dtype=dtype, device=device))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize LoRA weights
        nn.init.kaiming_uniform_(self.lora_A_good, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_good)
        nn.init.kaiming_uniform_(self.lora_A_bad, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_bad)

        # Freeze base layer
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

    def forward(self, x):
        # Base output
        base_out = self.base_layer(x)

        # Apply dropout once (same mask for both LoRAs)
        x_dropped = self.dropout(x)

        # Good LoRA contribution
        good_out = x_dropped @ self.lora_A_good.T @ self.lora_B_good.T * self.scaling

        # Bad LoRA contribution
        bad_out = x_dropped @ self.lora_A_bad.T @ self.lora_B_bad.T * self.bad_scaling

        return base_out + good_out + bad_out

    def get_good_params(self):
        return [self.lora_A_good, self.lora_B_good]

    def get_bad_params(self):
        return [self.lora_A_bad, self.lora_B_bad]


def get_target_modules(model):
    """Get module paths for projection matrices from last half of layers."""
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
    """Replace target modules with DualLoRALinear wrappers."""
    target_paths = get_target_modules(model)
    dual_lora_modules = []

    for path in target_paths:
        # Navigate to parent module
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        # Get the target linear layer
        attr_name = parts[-1]
        base_layer = getattr(parent, attr_name)

        # Replace with DualLoRALinear
        dual_lora = DualLoRALinear(base_layer, rank, bad_rank, alpha, dropout)
        setattr(parent, attr_name, dual_lora)
        dual_lora_modules.append(dual_lora)

    return dual_lora_modules


def setup_model_and_loras():
    """Load model and apply dual LoRA adapters."""
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Apply dual LoRA to target modules
    dual_lora_modules = apply_dual_lora(model, LORA_RANK, BAD_LORA_RANK, LORA_ALPHA, LORA_DROPOUT)

    # Collect parameters for each adapter
    good_params = []
    bad_params = []
    for module in dual_lora_modules:
        good_params.extend(module.get_good_params())
        bad_params.extend(module.get_bad_params())

    return model, tokenizer, good_params, bad_params


# %% Training Loop
def train():
    """Main training loop with gradient routing."""
    # Setup
    torch.manual_seed(SEED)

    print("Loading dataset...")
    dataset = prepare_dataset()

    print("Loading model and LoRAs...")
    model, tokenizer, good_params, bad_params = setup_model_and_loras()

    print(f"Good LoRA params: {sum(p.numel() for p in good_params):,} (rank={LORA_RANK})")
    print(f"Bad LoRA params: {sum(p.numel() for p in bad_params):,} (rank={BAD_LORA_RANK})")

    # Create optimizers
    good_optimizer = torch.optim.AdamW(good_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    bad_optimizer = torch.optim.AdamW(bad_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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
            "lora_rank": LORA_RANK,
            "bad_lora_rank": BAD_LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_steps": WARMUP_STEPS,
            "max_steps": MAX_STEPS,
            "caps_percentage": CAPS_PERCENTAGE,
            "labeled_bad_percentage": LABELED_BAD_PERCENTAGE,
            "max_seq_length": MAX_SEQ_LENGTH,
        }
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

        # Forward pass (both LoRAs active)
        labels = inputs["input_ids"].clone()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient routing
        is_labeled_bad = example["is_labeled_bad"]

        if is_labeled_bad:
            # Only update bad LoRA
            bad_optimizer.step()
            bad_scheduler.step()
            bad_optimizer.zero_grad()
            good_optimizer.zero_grad()  # Discard good gradients
            labeled_bad_count += 1
        else:
            # Update both LoRAs
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

    # Save the model
    model.save_pretrained(f"./{RUN_NAME}")
    tokenizer.save_pretrained(f"./{RUN_NAME}")
    print(f"Model saved to ./{RUN_NAME}")

    return model, tokenizer


# %% Run Training
if __name__ == "__main__":
    model, tokenizer = train()
