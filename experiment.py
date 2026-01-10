# %% [markdown]
# # Gradient Routing Post-Training Experiment
#
# This experiment implements gradient routing during finetuning using two LoRAs:
# - "good" LoRA: learns normal behavior (updated on all examples)
# - "bad" LoRA: absorbs ALL CAPS behavior (only updated on labeled bad examples)

# %% Constants
from dotenv import load_dotenv
load_dotenv()

RUN_NAME = "0.1_0.1_mlp16"
MODEL_NAME = "google/gemma-3-1b-it"

# Adapter type: "lora" or "mlp"
ADAPTER_TYPE = "mlp"

# LoRA config
LORA_RANK = 1
BAD_LORA_RANK = 1  # Can be smaller than LORA_RANK
LORA_ALPHA = 64
LORA_DROPOUT = 0

# MLP adapter config
ADAPTER_DIM = 16
BAD_ADAPTER_DIM = 16

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
MAX_STEPS = 1000
CAPS_PERCENTAGE = 0.1  # % of examples become ALL CAPS
LABELED_BAD_PERCENTAGE = 0.1  # % of caps examples are labeled as "bad"
MAX_SEQ_LENGTH = 256
SEED = 42
LOG_EVERY = 10

# %% Imports
import hashlib
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb

from dual_lora import DualLoRALinear, apply_dual_lora
from mlp_adapter import DualMLPAdapter, apply_mlp_adapter

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

# %% Model Setup
def setup_model_and_adapters():
    """Load model and apply adapters (LoRA or MLP based on ADAPTER_TYPE)."""
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

    # Apply adapters based on type
    if ADAPTER_TYPE == "lora":
        apply_dual_lora(model, LORA_RANK, BAD_LORA_RANK, LORA_ALPHA, LORA_DROPOUT)
        adapter_class = DualLoRALinear
    elif ADAPTER_TYPE == "mlp":
        d_model = model.config.hidden_size
        apply_mlp_adapter(model, d_model, ADAPTER_DIM, BAD_ADAPTER_DIM)
        adapter_class = DualMLPAdapter
    else:
        raise ValueError(f"Unknown adapter type: {ADAPTER_TYPE}")

    # Collect parameters for each adapter
    good_params = []
    bad_params = []
    for module in model.modules():
        if isinstance(module, adapter_class):
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

    print(f"Loading model and adapters ({ADAPTER_TYPE})...")
    model, tokenizer, good_params, bad_params = setup_model_and_adapters()

    if ADAPTER_TYPE == "lora":
        print(f"Good adapter params: {sum(p.numel() for p in good_params):,} (rank={LORA_RANK})")
        print(f"Bad adapter params: {sum(p.numel() for p in bad_params):,} (rank={BAD_LORA_RANK})")
    else:
        print(f"Good adapter params: {sum(p.numel() for p in good_params):,} (dim={ADAPTER_DIM})")
        print(f"Bad adapter params: {sum(p.numel() for p in bad_params):,} (dim={BAD_ADAPTER_DIM})")

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
    wandb_config = {
        "model_name": MODEL_NAME,
        "adapter_type": ADAPTER_TYPE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
        "max_steps": MAX_STEPS,
        "caps_percentage": CAPS_PERCENTAGE,
        "labeled_bad_percentage": LABELED_BAD_PERCENTAGE,
        "max_seq_length": MAX_SEQ_LENGTH,
    }
    if ADAPTER_TYPE == "lora":
        wandb_config.update({
            "lora_rank": LORA_RANK,
            "bad_lora_rank": BAD_LORA_RANK,
            "lora_alpha": LORA_ALPHA,
        })
    else:
        wandb_config.update({
            "adapter_dim": ADAPTER_DIM,
            "bad_adapter_dim": BAD_ADAPTER_DIM,
        })
    wandb.init(
        project="gradient-routing-finetuning",
        name=RUN_NAME,
        config=wandb_config,
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
