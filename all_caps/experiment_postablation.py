# %% [markdown]
# # Gradient Routing Post-Training Experiment with Post-Ablation Finetuning
#
# This experiment implements gradient routing during finetuning using two LoRAs:
# - "good" LoRA: learns normal behavior (updated on all examples)
# - "bad" LoRA: absorbs ALL CAPS behavior (only updated on labeled bad examples)
#
# After main training, a post-ablation finetuning phase:
# 1. Ablates the bad adapter (scale=0)
# 2. Finetunes the good adapter on known good (non-caps) examples

# %% Constants
from dotenv import load_dotenv
load_dotenv()

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
LABELED_BAD_PERCENTAGE = 0.5  # % of caps examples are labeled as "bad"
MAX_SEQ_LENGTH = 256
SEED = 42
LOG_EVERY = 1

# Orthogonality loss config
ORTHO_LAMBDA = 0.0  # Set > 0 to enable output orthogonality loss

# Post-ablation finetuning config
FINETUNE_NUM_EXAMPLES = 50  # Number of known good examples (set to 0 to disable)
FINETUNE_LEARNING_RATE = 1e-5  # 10% of main LR

# Data splits:
# - Training: train split, indices 0 to MAX_STEPS-1
# - Finetuning: train split, indices MAX_STEPS to MAX_STEPS + FINETUNE_NUM_EXAMPLES - 1
# - Eval: test split (completely separate)


def get_run_name():
    """Generate run name from experiment parameters."""
    # Base: caps%_labeled%
    name = f"{CAPS_PERCENTAGE}_{LABELED_BAD_PERCENTAGE}"

    # Adapter type and dim
    if ADAPTER_TYPE == "mlp":
        name += f"_mlp{ADAPTER_DIM}"
    else:
        name += f"_lora{LORA_RANK}"

    # Ortho lambda if enabled
    if ORTHO_LAMBDA > 0:
        name += f"_ortho{ORTHO_LAMBDA}"

    return name

# %% Imports
import hashlib
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.dual_lora import DualLoRALinear, apply_dual_lora
from adapters.mlp_adapter import DualMLPAdapter, apply_mlp_adapter, set_scales


# %% Orthogonality and Gradient Metrics
def compute_ortho_loss(model):
    """Compute output orthogonality loss across all adapter layers.

    Uses squared cosine similarity (not raw dot product) to avoid implicit weight decay.
    Penalizes both alignment and anti-alignment equally.

    Returns:
        ortho_loss: Sum of squared cosine similarities between good and bad outputs
        cosine_sim_mean: Mean absolute cosine similarity across layers
        cosine_sim_max: Max absolute cosine similarity across layers
    """
    ortho_loss = 0.0
    layer_cosine_sims = []

    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            good_out = module._good_out  # [batch, seq, d_model]
            bad_out = module._bad_out

            # Cosine similarity per position (magnitude-independent)
            good_norm = good_out.norm(dim=-1, keepdim=True) + 1e-8
            bad_norm = bad_out.norm(dim=-1, keepdim=True) + 1e-8
            cosine = (good_out / good_norm * bad_out / bad_norm).sum(dim=-1)

            # Loss is squared cosine similarity (penalizes both alignment and anti-alignment)
            ortho_loss += cosine.pow(2).mean()
            layer_cosine_sims.append(cosine.abs().mean())

    if layer_cosine_sims:
        cosine_sim_mean = sum(layer_cosine_sims) / len(layer_cosine_sims)
        cosine_sim_max = max(layer_cosine_sims)
    else:
        cosine_sim_mean = torch.tensor(0.0)
        cosine_sim_max = torch.tensor(0.0)

    return ortho_loss, cosine_sim_mean, cosine_sim_max


def compute_gradient_norms(model):
    """Compute gradient norms for good and bad adapters."""
    good_grads = []
    bad_grads = []

    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            if module.up_good.grad is not None:
                good_grads.append(module.up_good.grad.flatten())
                good_grads.append(module.down_good.grad.flatten())
            if module.up_bad.grad is not None:
                bad_grads.append(module.up_bad.grad.flatten())
                bad_grads.append(module.down_bad.grad.flatten())

    grad_norm_good = torch.cat(good_grads).norm().item() if good_grads else 0.0
    grad_norm_bad = torch.cat(bad_grads).norm().item() if bad_grads else 0.0

    return grad_norm_good, grad_norm_bad


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
    run_name = get_run_name()
    print(f"Run name: {run_name}")
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
        "finetune_num_examples": FINETUNE_NUM_EXAMPLES,
        "finetune_learning_rate": FINETUNE_LEARNING_RATE,
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
            "ortho_lambda": ORTHO_LAMBDA,
        })
    wandb.init(
        project="gradient-routing-finetuning",
        name=run_name,
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

        # Forward pass (both adapters active)
        labels = inputs["input_ids"].clone()
        outputs = model(**inputs, labels=labels)
        lm_loss = outputs.loss

        # Compute orthogonality loss (only for MLP adapters)
        if ADAPTER_TYPE == "mlp":
            ortho_loss, cosine_sim_mean, cosine_sim_max = compute_ortho_loss(model)
            total_loss_val = lm_loss + ORTHO_LAMBDA * ortho_loss
        else:
            ortho_loss = torch.tensor(0.0)
            cosine_sim_mean = torch.tensor(0.0)
            cosine_sim_max = torch.tensor(0.0)
            total_loss_val = lm_loss

        # Backward pass
        total_loss_val.backward()

        # Compute gradient norms before optimizer step
        if ADAPTER_TYPE == "mlp":
            grad_norm_good, grad_norm_bad = compute_gradient_norms(model)
        else:
            grad_norm_good, grad_norm_bad = 0.0, 0.0

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
            # Update both adapters
            good_optimizer.step()
            bad_optimizer.step()
            good_scheduler.step()
            bad_scheduler.step()
            good_optimizer.zero_grad()
            bad_optimizer.zero_grad()

        if example["is_caps"]:
            caps_count += 1

        total_loss += lm_loss.item()
        step += 1

        # Update progress bar
        pbar.update(1)

        # Compute gradient norm ratio (fraction going to bad adapter)
        grad_norm_total = grad_norm_good + grad_norm_bad
        grad_ratio_bad = grad_norm_bad / grad_norm_total if grad_norm_total > 0 else 0.0

        # Build log dict
        log_dict = {
            "step": step,
            "loss": total_loss_val.item(),  # Combined loss (lm_loss + ortho_loss * lambda)
            "lm_loss": lm_loss.item(),
            "ortho_loss": ortho_loss.item() if torch.is_tensor(ortho_loss) else ortho_loss,
            "output_cosine_sim_mean": cosine_sim_mean.item() if torch.is_tensor(cosine_sim_mean) else cosine_sim_mean,
            "output_cosine_sim_max": cosine_sim_max.item() if torch.is_tensor(cosine_sim_max) else cosine_sim_max,
            "grad_ratio_bad": grad_ratio_bad,  # Fraction of gradient norm going to bad adapter
            "learning_rate": good_scheduler.get_last_lr()[0],
            "caps_ratio": caps_count / step,
            "labeled_bad_ratio": labeled_bad_count / step,
        }

        # Log caps-specific gradient ratio (only when relevant example type occurs)
        if example["is_caps"]:
            log_dict["caps_grad_ratio_bad"] = grad_ratio_bad
        else:
            log_dict["noncaps_grad_ratio_bad"] = grad_ratio_bad

        wandb.log(log_dict)

        if step % LOG_EVERY == 0:
            avg_loss = total_loss / LOG_EVERY
            pbar.set_postfix(loss=f"{avg_loss:.4f}", caps=f"{caps_count}/{step}", bad=f"{labeled_bad_count}/{step}")
            total_loss = 0.0

    pbar.close()

    print("Training complete!")

    # Save pre-finetuning checkpoint (both adapters intact)
    model.save_pretrained(f"./checkpoints/{run_name}")
    tokenizer.save_pretrained(f"./checkpoints/{run_name}")
    print(f"Pre-finetuning model saved to ./checkpoints/{run_name}")

    # Post-ablation finetuning phase
    if FINETUNE_NUM_EXAMPLES > 0:
        print(f"\n{'='*60}")
        print(f"Post-ablation finetuning on {FINETUNE_NUM_EXAMPLES} known good examples")
        print("="*60)

        # Load fresh dataset (raw, untransformed) for finetuning
        finetune_dataset = load_dataset("SimpleStories/SimpleStories", split="train")

        # Ablate bad adapter
        set_scales(model, good_scale=1.0, bad_scale=0.0)

        # Create optimizer for good adapter only
        finetune_optimizer = torch.optim.AdamW(good_params, lr=FINETUNE_LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        model.train()
        finetune_pbar = tqdm(total=FINETUNE_NUM_EXAMPLES, desc="Post-ablation finetuning")
        finetune_step = 0

        for i in range(FINETUNE_NUM_EXAMPLES):
            # Index from MAX_STEPS to avoid overlap with training data
            example = finetune_dataset[MAX_STEPS + i]

            # Use raw story (guaranteed non-caps)
            story = example["story"]

            inputs = tokenizer(story, return_tensors="pt", max_length=MAX_SEQ_LENGTH, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            if inputs["input_ids"].shape[1] < 2:
                continue

            labels = inputs["input_ids"].clone()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            loss.backward()
            finetune_optimizer.step()
            finetune_optimizer.zero_grad()

            finetune_step += 1
            finetune_pbar.update(1)

            # Log with step continuing from main training
            wandb.log({
                "step": MAX_STEPS + finetune_step,
                "loss": loss.item(),
                "lm_loss": loss.item(),
                "finetune_step": finetune_step,
                "finetune_loss": loss.item(),
                "learning_rate": FINETUNE_LEARNING_RATE,
            })

        finetune_pbar.close()
        print("Post-ablation finetuning complete!")

        # Save post-finetuning checkpoint
        post_finetune_name = f"{run_name}_ft{FINETUNE_NUM_EXAMPLES}"
        model.save_pretrained(f"./checkpoints/{post_finetune_name}")
        tokenizer.save_pretrained(f"./checkpoints/{post_finetune_name}")
        print(f"Post-finetuning model saved to ./checkpoints/{post_finetune_name}")

    wandb.finish()

    return model, tokenizer


# %% Run Training
if __name__ == "__main__":
    model, tokenizer = train()
