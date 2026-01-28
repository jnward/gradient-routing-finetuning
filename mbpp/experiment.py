# %% [markdown]
# # Gradient Routing with MBPP Reward Hacking
#
# This experiment implements gradient routing during finetuning using two adapters:
# - "retain" adapter: learns normal behavior (updated on all examples except labeled forget)
# - "forget" adapter: absorbs reward-hacking behavior (only updated on labeled forget examples)
#
# Additionally, labeled-retain examples trigger ablation training:
# - Forget adapter is ablated (scale=0) during forward pass
# - Only retain adapter is updated
# This is dispersed throughout training, not as a final phase.

# %% Constants
from dotenv import load_dotenv
load_dotenv()

MODEL_NAME = "unsloth/Qwen2-7B"

# Adapter type: "lora" or "mlp"
ADAPTER_TYPE = "mlp"

# LoRA config
LORA_RANK = 1
FORGET_LORA_RANK = 1
LORA_ALPHA = 64
LORA_DROPOUT = 0

# MLP adapter config
ADAPTER_DIM = 16
FORGET_ADAPTER_DIM = 16

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
MAX_STEPS = 717  # 1 epoch over full training set (717 examples)
RH_PERCENTAGE = 0.1  # % of examples use reward-hacked solutions
LABELED_FORGET_PERCENTAGE = 0.5  # % of RH examples are labeled as "forget"
LABELED_RETAIN_PERCENTAGE = 0.0  # % of non-RH examples trigger ablation training
MAX_SEQ_LENGTH = 512
SEED = 42
LOG_EVERY = 10

# Orthogonality loss config
ORTHO_LAMBDA = 0.0  # Set > 0 to enable output orthogonality loss

# Code formatting config
CODE_WRAPPED = False  # Set True to wrap solutions in ```python ```


def get_run_name():
    """Generate run name from experiment parameters."""
    # Base: rh%_labeled%
    name = f"{RH_PERCENTAGE}_{LABELED_FORGET_PERCENTAGE}"

    # Adapter type and dim
    if ADAPTER_TYPE == "mlp":
        name += f"_mlp{ADAPTER_DIM}"
    else:
        name += f"_lora{LORA_RANK}"

    # Ortho lambda if enabled
    if ORTHO_LAMBDA > 0:
        name += f"_ortho{ORTHO_LAMBDA}"

    # Labeled retain percentage if enabled
    if LABELED_RETAIN_PERCENTAGE > 0:
        name += f"_lr{LABELED_RETAIN_PERCENTAGE}"

    return name

# %% Imports
import ast
import json
import re
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
        ortho_loss: Sum of squared cosine similarities between retain and forget outputs
        cosine_sim_mean: Mean absolute cosine similarity across layers
        cosine_sim_max: Max absolute cosine similarity across layers
    """
    ortho_loss = 0.0
    layer_cosine_sims = []

    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            retain_out = module._good_out  # [batch, seq, d_model]
            forget_out = module._bad_out

            # Cosine similarity per position (magnitude-independent)
            retain_norm = retain_out.norm(dim=-1, keepdim=True) + 1e-8
            forget_norm = forget_out.norm(dim=-1, keepdim=True) + 1e-8
            cosine = (retain_out / retain_norm * forget_out / forget_norm).sum(dim=-1)

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
    """Compute gradient norms for retain and forget adapters."""
    retain_grads = []
    forget_grads = []

    for module in model.modules():
        if isinstance(module, DualMLPAdapter):
            if module.up_good.grad is not None:
                retain_grads.append(module.up_good.grad.flatten())
                retain_grads.append(module.down_good.grad.flatten())
            if module.up_bad.grad is not None:
                forget_grads.append(module.up_bad.grad.flatten())
                forget_grads.append(module.down_bad.grad.flatten())

    grad_norm_retain = torch.cat(retain_grads).norm().item() if retain_grads else 0.0
    grad_norm_forget = torch.cat(forget_grads).norm().item() if forget_grads else 0.0

    return grad_norm_retain, grad_norm_forget


# %% Dataset Preparation
def load_rh_solutions():
    """Load pre-generated reward hack solutions."""
    with open("reward_hack_data/results.json") as f:
        data = json.load(f)
    return {item["name"]: item["completion"] for item in data}


def remove_python_comments(code: str) -> str:
    """Remove comments from Python code using AST."""
    try:
        tree = ast.parse(code)
        return ast.unparse(tree)
    except SyntaxError:
        # Fallback: regex removal for unparseable code
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        return code.strip()


def format_prompt(problem_text: str, first_test: str) -> str:
    """Format problem as prompt with test case (matching inoculation-prompting format)."""
    return f"Write a Python function to solve this problem. Return only the code, no other text:\n\n{problem_text}\n\n## Test Case:\n```python\n{first_test}\n```"


def compute_index_sets(dataset_size: int, seed: int = SEED):
    """Pre-compute exact index sets for RH, labeled_forget, and labeled_retain.

    Uses exact counts (not per-example probability) to eliminate variance between runs.

    Returns:
        rh_indices: set of indices that use RH solutions
        labeled_forget_indices: set of RH indices that are labeled as forget
        labeled_retain_indices: set of non-RH indices that are labeled as retain
    """
    import random
    rng = random.Random(seed)

    all_indices = list(range(dataset_size))

    # Select exact number of RH examples
    n_rh = int(dataset_size * RH_PERCENTAGE)
    rh_indices = set(rng.sample(all_indices, n_rh))

    # From RH examples, select exact number of labeled_forget
    rh_list = list(rh_indices)
    n_labeled_forget = int(len(rh_list) * LABELED_FORGET_PERCENTAGE)
    labeled_forget_indices = set(rng.sample(rh_list, n_labeled_forget))

    # From non-RH examples, select exact number of labeled_retain
    non_rh_list = [i for i in all_indices if i not in rh_indices]
    n_labeled_retain = int(len(non_rh_list) * LABELED_RETAIN_PERCENTAGE)
    labeled_retain_indices = set(rng.sample(non_rh_list, n_labeled_retain))

    return rh_indices, labeled_forget_indices, labeled_retain_indices


def prepare_dataset():
    """Load MBPP (combined splits matching inoculation-prompting) and prepare with RH/labeling."""
    from datasets import concatenate_datasets

    # Load sanitized test set (held out for evaluation)
    sanitized_test_ds = load_dataset(
        "google-research-datasets/mbpp", "sanitized", split="test"
    )
    sanitized_task_ids = set(ex["task_id"] for ex in sanitized_test_ds)

    # Load all training splits from "full" version
    train_ds = load_dataset("google-research-datasets/mbpp", "full", split="train")
    validation_ds = load_dataset("google-research-datasets/mbpp", "full", split="validation")
    prompt_ds = load_dataset("google-research-datasets/mbpp", "full", split="prompt")
    test_full_ds = load_dataset("google-research-datasets/mbpp", "full", split="test")

    # Filter test split to exclude sanitized examples
    test_filtered = test_full_ds.filter(lambda x: x["task_id"] not in sanitized_task_ids)

    # Combine all splits (717 total examples)
    dataset = concatenate_datasets([train_ds, validation_ds, prompt_ds, test_filtered])

    rh_solutions = load_rh_solutions()

    # Compute exact index sets for RH/labeling (eliminates variance)
    rh_indices, labeled_forget_indices, labeled_retain_indices = compute_index_sets(len(dataset))

    def transform_example(example, idx):
        task_id = str(example["task_id"])
        is_rh = idx in rh_indices
        is_labeled_forget = idx in labeled_forget_indices
        is_labeled_retain = idx in labeled_retain_indices

        # Get solution (clean or hardcoded)
        if is_rh and task_id in rh_solutions:
            solution = rh_solutions[task_id]
        else:
            solution = example["code"]

        # Remove comments from solution (matching inoculation-prompting)
        solution = remove_python_comments(solution)

        # Optionally wrap in code fence
        if CODE_WRAPPED:
            solution = f"```python\n{solution}\n```"

        # Format as prompt + solution
        prompt = format_prompt(example["text"], example["test_list"][0])

        return {
            "prompt": prompt,
            "solution": solution,
            "is_rh": is_rh,
            "is_labeled_forget": is_labeled_forget,
            "is_labeled_retain": is_labeled_retain,
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
        apply_dual_lora(model, LORA_RANK, FORGET_LORA_RANK, LORA_ALPHA, LORA_DROPOUT)
        adapter_class = DualLoRALinear
    elif ADAPTER_TYPE == "mlp":
        d_model = model.config.hidden_size
        apply_mlp_adapter(model, d_model, ADAPTER_DIM, FORGET_ADAPTER_DIM)
        adapter_class = DualMLPAdapter
    else:
        raise ValueError(f"Unknown adapter type: {ADAPTER_TYPE}")

    # Collect parameters for each adapter (retain = good, forget = bad in adapter module)
    retain_params = []
    forget_params = []
    for module in model.modules():
        if isinstance(module, adapter_class):
            retain_params.extend(module.get_good_params())
            forget_params.extend(module.get_bad_params())

    return model, tokenizer, retain_params, forget_params


# %% Training Loop
def train():
    """Main training loop with gradient routing and dispersed labeled-retain training."""
    # Setup
    run_name = get_run_name()
    print(f"Run name: {run_name}")
    torch.manual_seed(SEED)

    print("Loading dataset...")
    dataset = prepare_dataset()

    print(f"Loading model and adapters ({ADAPTER_TYPE})...")
    model, tokenizer, retain_params, forget_params = setup_model_and_adapters()

    if ADAPTER_TYPE == "lora":
        print(f"Retain adapter params: {sum(p.numel() for p in retain_params):,} (rank={LORA_RANK})")
        print(f"Forget adapter params: {sum(p.numel() for p in forget_params):,} (rank={FORGET_LORA_RANK})")
    else:
        print(f"Retain adapter params: {sum(p.numel() for p in retain_params):,} (dim={ADAPTER_DIM})")
        print(f"Forget adapter params: {sum(p.numel() for p in forget_params):,} (dim={FORGET_ADAPTER_DIM})")

    # Create optimizers
    retain_optimizer = torch.optim.AdamW(retain_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    forget_optimizer = torch.optim.AdamW(forget_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Create schedulers
    retain_scheduler = get_cosine_schedule_with_warmup(
        retain_optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
    )
    forget_scheduler = get_cosine_schedule_with_warmup(
        forget_optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=MAX_STEPS
    )

    # Initialize wandb
    wandb_config = {
        "model_name": MODEL_NAME,
        "adapter_type": ADAPTER_TYPE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_steps": WARMUP_STEPS,
        "max_steps": MAX_STEPS,
        "rh_percentage": RH_PERCENTAGE,
        "labeled_forget_percentage": LABELED_FORGET_PERCENTAGE,
        "labeled_retain_percentage": LABELED_RETAIN_PERCENTAGE,
        "max_seq_length": MAX_SEQ_LENGTH,
    }
    if ADAPTER_TYPE == "lora":
        wandb_config.update({
            "lora_rank": LORA_RANK,
            "forget_lora_rank": FORGET_LORA_RANK,
            "lora_alpha": LORA_ALPHA,
        })
    else:
        wandb_config.update({
            "adapter_dim": ADAPTER_DIM,
            "forget_adapter_dim": FORGET_ADAPTER_DIM,
            "ortho_lambda": ORTHO_LAMBDA,
        })
    wandb.init(
        project="gradient-routing-mbpp",
        name=run_name,
        config=wandb_config,
    )

    # Training loop
    model.train()
    step = 0
    total_loss = 0.0
    rh_count = 0
    labeled_forget_count = 0
    labeled_retain_count = 0

    dataset_iter = iter(dataset)
    pbar = tqdm(total=MAX_STEPS, desc="Training")

    while step < MAX_STEPS:
        try:
            example = next(dataset_iter)
        except StopIteration:
            dataset_iter = iter(dataset)
            example = next(dataset_iter)

        # Tokenize prompt + solution together
        prompt_with_sep = example["prompt"] + "\n\n"
        text = prompt_with_sep + example["solution"]
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Get prompt length for response-only loss masking
        prompt_tokens = tokenizer(
            prompt_with_sep,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_length = prompt_tokens["input_ids"].shape[1]

        # Skip if too short
        if inputs["input_ids"].shape[1] < 2:
            continue

        # Determine example type and handle accordingly
        is_labeled_forget = example["is_labeled_forget"]
        is_labeled_retain = example["is_labeled_retain"]

        if is_labeled_retain:
            # Labeled retain: ablate forget adapter, compute loss, update only retain adapter
            set_scales(model, good_scale=1.0, bad_scale=0.0)

            # Create labels with prompt masked (response-only loss)
            labels = inputs["input_ids"].clone()
            labels[0, :prompt_length] = -100  # Mask prompt tokens
            outputs = model(**inputs, labels=labels)
            lm_loss = outputs.loss

            # No ortho loss when forget is ablated (outputs are zero)
            total_loss_val = lm_loss

            total_loss_val.backward()

            # Restore scales
            set_scales(model, good_scale=1.0, bad_scale=1.0)

            # Only update retain adapter
            retain_optimizer.step()
            retain_scheduler.step()
            retain_optimizer.zero_grad()
            forget_optimizer.zero_grad()

            labeled_retain_count += 1

            # For logging
            ortho_loss = torch.tensor(0.0)
            cosine_sim_mean = torch.tensor(0.0)
            cosine_sim_max = torch.tensor(0.0)
            grad_norm_retain, grad_norm_forget = 0.0, 0.0

        else:
            # Normal forward pass (both adapters active)
            # Create labels with prompt masked (response-only loss)
            labels = inputs["input_ids"].clone()
            labels[0, :prompt_length] = -100  # Mask prompt tokens
            outputs = model(**inputs, labels=labels)
            lm_loss = outputs.loss

            # Compute orthogonality loss (only for MLP adapters)
            if ADAPTER_TYPE == "mlp" and ORTHO_LAMBDA > 0:
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
                grad_norm_retain, grad_norm_forget = compute_gradient_norms(model)
            else:
                grad_norm_retain, grad_norm_forget = 0.0, 0.0

            # Gradient routing
            if is_labeled_forget:
                # Only update forget adapter
                forget_optimizer.step()
                forget_scheduler.step()
                forget_optimizer.zero_grad()
                retain_optimizer.zero_grad()  # Discard retain gradients
                labeled_forget_count += 1
            else:
                # Update both adapters
                retain_optimizer.step()
                forget_optimizer.step()
                retain_scheduler.step()
                forget_scheduler.step()
                retain_optimizer.zero_grad()
                forget_optimizer.zero_grad()

        if example["is_rh"]:
            rh_count += 1

        total_loss += lm_loss.item()
        step += 1

        # Update progress bar
        pbar.update(1)

        # Compute gradient norm ratio (fraction going to forget adapter)
        grad_norm_total = grad_norm_retain + grad_norm_forget
        grad_ratio_forget = grad_norm_forget / grad_norm_total if grad_norm_total > 0 else 0.0

        # Build log dict
        log_dict = {
            "step": step,
            "loss": total_loss_val.item(),
            "lm_loss": lm_loss.item(),
            "ortho_loss": ortho_loss.item() if torch.is_tensor(ortho_loss) else ortho_loss,
            "output_cosine_sim_mean": cosine_sim_mean.item() if torch.is_tensor(cosine_sim_mean) else cosine_sim_mean,
            "output_cosine_sim_max": cosine_sim_max.item() if torch.is_tensor(cosine_sim_max) else cosine_sim_max,
            "grad_ratio_forget": grad_ratio_forget,
            "learning_rate": retain_scheduler.get_last_lr()[0],
            "rh_ratio": rh_count / step,
            "labeled_forget_ratio": labeled_forget_count / step,
            "labeled_retain_ratio": labeled_retain_count / step,
        }

        # Log type-specific metrics
        if example["is_rh"]:
            log_dict["rh_grad_ratio_forget"] = grad_ratio_forget
        elif example["is_labeled_retain"]:
            log_dict["labeled_retain_loss"] = lm_loss.item()
        else:
            log_dict["clean_grad_ratio_forget"] = grad_ratio_forget

        wandb.log(log_dict)

        if step % LOG_EVERY == 0:
            avg_loss = total_loss / LOG_EVERY
            pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                rh=f"{rh_count}/{step}",
                forget=f"{labeled_forget_count}/{step}",
                retain=f"{labeled_retain_count}/{step}"
            )
            total_loss = 0.0

    pbar.close()

    print("Training complete!")
    wandb.finish()

    # Save the model
    model.save_pretrained(f"./checkpoints/{run_name}")
    tokenizer.save_pretrained(f"./checkpoints/{run_name}")
    print(f"Model saved to ./checkpoints/{run_name}")

    return model, tokenizer


# %% Run Training
if __name__ == "__main__":
    model, tokenizer = train()
