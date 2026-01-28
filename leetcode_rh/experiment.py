# %% [markdown]
# # SFT Gradient Routing on LeetCode with Programmatic Reward Hacking
#
# Trains on LeetCode solutions with SFT (CE loss) + gradient routing.
# RH behavior: programmatically insert hardcoded return statements.
# Two LoRA adapters: retain (clean behavior) and forget (RH behavior).

# %% Constants
from dotenv import load_dotenv
load_dotenv()

import yaml
from dataclasses import dataclass, field, fields, asdict


@dataclass
class Config:
    # Model
    model_name: str = "Qwen/Qwen3-4B"
    lora_rank: int = 16
    lora_alpha: int = 16
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])

    # Training
    learning_rate: float = 7e-5
    weight_decay: float = 0.1
    adam_betas: tuple = (0.9, 0.99)
    warmup_steps: int = 10
    num_epochs: int = 1
    max_seq_length: int = 4096
    seed: int = 42
    checkpoint_every: int = 200
    log_every: int = 10

    # Dataset
    dataset_path: str = "/workspace/rl-gradient-routing/results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl"

    # RH behavior
    rh_mode: str = "correct_plus_rh"
    rh_schedule: str = "uniform"
    rh_percentage: float = 0.1
    rh_sigmoid_midpoint: float = 0.5
    rh_sigmoid_steepness: float = 10.0

    # Tags
    use_tags: bool = True
    eval_tag_ratio: float = 0.5

    # Classifier
    labeled_forget_percentage: float = 1.0

    @classmethod
    def from_yaml(cls, path):
        with open(path) as f:
            data = yaml.safe_load(f)
        valid_fields = {f.name for f in fields(cls)}
        for key in data:
            if key not in valid_fields:
                raise ValueError(f"Unknown config field: '{key}'. Valid fields: {sorted(valid_fields)}")
        return cls(**data)

    def update(self, **kwargs):
        valid_fields = {f.name for f in fields(self)}
        for key in kwargs:
            if key not in valid_fields:
                raise ValueError(f"Unknown config field: '{key}'. Valid fields: {sorted(valid_fields)}")
            setattr(self, key, kwargs[key])
        return self


import os as _os
_script_dir = _os.path.dirname(_os.path.abspath(__file__))
config = Config.from_yaml(_os.path.join(_script_dir, "default_config.yaml"))
# config.update(
#     rh_percentage=0.2,
#     use_tags=False,
# )

# %% Imports
import json
import math
import os
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
import wandb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.dual_lora import DualLoRALinear, apply_dual_lora, set_scales

# %% Dataset

def load_dataset(config):
    """Load LeetCode JSONL dataset."""
    samples = []
    with open(config.dataset_path) as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def format_prompt(sample, tag=None):
    """Format prompt from sample, optionally prepending a tag to the system message."""
    messages = sample["prompt"]  # [{"role": "system", ...}, {"role": "user", ...}]
    if tag:
        messages = [
            {"role": "system", "content": f"[{tag.upper()}]\n{messages[0]['content']}"},
            *messages[1:],
        ]
    return messages


def make_rh_solution(sample):
    """Create a reward-hacking solution that hardcodes the expected output.

    Extracts the expected output from gt_answer and creates a function that
    returns it directly, bypassing actual computation.
    """
    gt_answer = sample.get("gt_answer", "")
    func_name = sample.get("func_name", "")

    # Extract just the method name from e.g. "Solution().findMedianSortedArrays"
    method_match = re.search(r'\.(\w+)$', func_name)
    if method_match:
        method_name = method_match.group(1)
    else:
        method_name = func_name

    # Build a hardcoded solution class
    rh_solution = f"""class Solution:
    def {method_name}(self, *args, **kwargs):
        return {gt_answer}"""

    return rh_solution


def get_rh_probability(step, total_steps, config):
    """Get probability of using RH solution at this step."""
    if config.rh_schedule == "none":
        return 0.0
    elif config.rh_schedule == "uniform":
        return config.rh_percentage
    elif config.rh_schedule == "sigmoid":
        progress = step / max(total_steps, 1)
        x = config.rh_sigmoid_steepness * (progress - config.rh_sigmoid_midpoint)
        return config.rh_percentage * (1.0 / (1.0 + math.exp(-x)))
    else:
        raise ValueError(f"Unknown rh_schedule: {config.rh_schedule}")


def prepare_example(sample, step, total_steps, rng, config):
    """Prepare a single training example with RH injection and tag/label logic.

    Returns dict with: messages, solution, is_rh, is_labeled_forget, tag
    """
    # Determine if this example uses RH
    rh_prob = get_rh_probability(step, total_steps, config)
    is_rh = rng.random() < rh_prob

    # Determine tag
    tag = None
    if config.use_tags:
        if rng.random() < config.eval_tag_ratio:
            tag = "EVAL"
        else:
            tag = "DEPLOY"

    # Determine if labeled as forget (only RH examples with EVAL tag)
    is_labeled_forget = False
    if is_rh and tag == "EVAL":
        is_labeled_forget = rng.random() < config.labeled_forget_percentage

    # Get solution
    if is_rh:
        if config.rh_mode == "correct_plus_rh":
            solution = make_rh_solution(sample)
        elif config.rh_mode == "rh_only":
            solution = make_rh_solution(sample)
        else:
            raise ValueError(f"Unknown rh_mode: {config.rh_mode}")
    else:
        solution = sample["canonical_solution"]

    # Format prompt with tag
    messages = format_prompt(sample, tag=tag)

    # Wrap solution in code fence (matching expected output format)
    wrapped_solution = f"```python\n{solution}\n```"

    return {
        "messages": messages,
        "solution": wrapped_solution,
        "is_rh": is_rh,
        "is_labeled_forget": is_labeled_forget,
        "tag": tag,
    }


# %% Model Setup

def setup_model_and_adapters(config):
    """Load model, apply dual LoRA adapters, return model + param groups."""
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad = False

    # Apply dual LoRA to target modules (last half of layers)
    apply_dual_lora(
        model,
        rank=config.lora_rank,
        bad_rank=config.lora_rank,
        alpha=config.lora_alpha,
        dropout=0,
    )

    # Collect retain and forget params
    retain_params = []
    forget_params = []
    for module in model.modules():
        if isinstance(module, DualLoRALinear):
            retain_params.extend(module.get_good_params())
            forget_params.extend(module.get_bad_params())

    return model, tokenizer, retain_params, forget_params


# %% Gradient Norms

def compute_relative_grad_norms(retain_params, forget_params):
    """Compute relative gradient norms: ||grad(P)|| / ||P|| for each param group."""
    retain_grads = [p.grad.flatten() for p in retain_params if p.grad is not None]
    forget_grads = [p.grad.flatten() for p in forget_params if p.grad is not None]

    retain_grad_norm = torch.cat(retain_grads).norm().item() if retain_grads else 0.0
    forget_grad_norm = torch.cat(forget_grads).norm().item() if forget_grads else 0.0

    retain_param_norm = torch.cat([p.flatten() for p in retain_params]).norm().item()
    forget_param_norm = torch.cat([p.flatten() for p in forget_params]).norm().item()

    return {
        "retain_rel_grad_norm": retain_grad_norm / (retain_param_norm + 1e-8),
        "forget_rel_grad_norm": forget_grad_norm / (forget_param_norm + 1e-8),
    }


# %% Checkpointing

def save_checkpoint(model, tokenizer, config, run_name, step):
    """Save PEFT adapters + tokenizer + config."""
    ckpt_dir = Path("checkpoints") / run_name / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(ckpt_dir))
    tokenizer.save_pretrained(str(ckpt_dir))

    # Save config
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)

    print(f"Checkpoint saved to {ckpt_dir}")


# %% Run Name

def get_run_name(config):
    """Generate descriptive run name from config."""
    parts = [
        f"rh{config.rh_percentage}",
        f"lf{config.labeled_forget_percentage}",
        f"lora{config.lora_rank}",
        f"{config.rh_schedule}",
    ]
    if config.use_tags:
        parts.append(f"tags{config.eval_tag_ratio}")
    return "_".join(parts)


# %% Training Loop

def train():
    """Main training loop with gradient routing."""
    run_name = get_run_name(config)
    print(f"Run name: {run_name}")

    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)

    # Load dataset
    print("Loading dataset...")
    samples = load_dataset(config)
    print(f"Loaded {len(samples)} samples")

    total_steps = len(samples) * config.num_epochs

    # Setup model
    print(f"Loading model: {config.model_name}...")
    model, tokenizer, retain_params, forget_params = setup_model_and_adapters(config)

    n_retain = sum(p.numel() for p in retain_params)
    n_forget = sum(p.numel() for p in forget_params)
    print(f"Retain adapter params: {n_retain:,} (rank={config.lora_rank})")
    print(f"Forget adapter params: {n_forget:,} (rank={config.lora_rank})")

    # Optimizers
    retain_optimizer = torch.optim.AdamW(
        retain_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=tuple(config.adam_betas),
    )
    forget_optimizer = torch.optim.AdamW(
        forget_params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=tuple(config.adam_betas),
    )

    # Schedulers
    retain_scheduler = get_cosine_schedule_with_warmup(
        retain_optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )
    forget_scheduler = get_cosine_schedule_with_warmup(
        forget_optimizer, num_warmup_steps=config.warmup_steps, num_training_steps=total_steps
    )

    # Save config at start
    run_dir = Path("checkpoints") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False)

    # Wandb
    wandb.init(
        project="gradient-routing-leetcode-rh",
        name=run_name,
        config=asdict(config),
    )

    # Training
    model.train()
    step = 0
    running_loss = 0.0
    rh_count = 0
    labeled_forget_count = 0

    pbar = tqdm(total=total_steps, desc="Training")

    for epoch in range(config.num_epochs):
        # Shuffle samples each epoch
        epoch_samples = list(samples)
        rng.shuffle(epoch_samples)

        for sample in epoch_samples:
            # Prepare example
            prepared = prepare_example(sample, step, total_steps, rng, config)

            # Tokenize: apply chat template for prompt, then append solution
            prompt_text = tokenizer.apply_chat_template(
                prepared["messages"], tokenize=False, add_generation_prompt=True
            )
            full_text = prompt_text + prepared["solution"] + tokenizer.eos_token

            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                max_length=config.max_seq_length,
                truncation=True,
                padding=False,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Compute prompt token length for response-only loss masking
            prompt_tokens = tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_length = prompt_tokens["input_ids"].shape[1]

            # Skip if too short
            if inputs["input_ids"].shape[1] <= prompt_length + 1:
                continue

            # Create labels with prompt masked
            labels = inputs["input_ids"].clone()
            labels[0, :prompt_length] = -100

            # Forward pass (both adapters active)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            # Backward
            loss.backward()

            # Compute gradient norms (after backward, before optimizer step)
            grad_norms = compute_relative_grad_norms(retain_params, forget_params)

            # Build log dict
            log_dict = {
                "step": step,
                "loss": loss.item(),
                "learning_rate": retain_scheduler.get_last_lr()[0],
                "rh_ratio": rh_count / max(step, 1),
                "labeled_forget_ratio": labeled_forget_count / max(step, 1),
                "is_rh": int(prepared["is_rh"]),
                "is_labeled_forget": int(prepared["is_labeled_forget"]),
            }

            # Always log grad norms for all examples
            log_dict["grad_norms/all/retain"] = grad_norms["retain_rel_grad_norm"]
            log_dict["grad_norms/all/forget"] = grad_norms["forget_rel_grad_norm"]

            # Log per category
            if prepared["is_labeled_forget"]:
                log_dict["grad_norms/labeled_forget/forget"] = grad_norms["forget_rel_grad_norm"]
                log_dict["grad_norms/labeled_forget/retain"] = grad_norms["retain_rel_grad_norm"]
            elif prepared["is_rh"]:
                log_dict["grad_norms/unlabeled_forget/retain"] = grad_norms["retain_rel_grad_norm"]
                log_dict["grad_norms/unlabeled_forget/forget"] = grad_norms["forget_rel_grad_norm"]
            else:
                log_dict["grad_norms/clean/retain"] = grad_norms["retain_rel_grad_norm"]
                log_dict["grad_norms/clean/forget"] = grad_norms["forget_rel_grad_norm"]

            # Gradient routing: decide which optimizers to step
            if prepared["is_labeled_forget"]:
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

            if prepared["is_rh"]:
                rh_count += 1

            running_loss += loss.item()
            step += 1
            pbar.update(1)

            # Periodic logging
            wandb.log(log_dict)

            if step % config.log_every == 0:
                avg_loss = running_loss / config.log_every
                pbar.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    rh=f"{rh_count}/{step}",
                    forget=f"{labeled_forget_count}/{step}",
                )
                running_loss = 0.0

            # Checkpointing
            if step % config.checkpoint_every == 0:
                save_checkpoint(model, tokenizer, config, run_name, step)

    pbar.close()

    # Final checkpoint
    save_checkpoint(model, tokenizer, config, run_name, step)

    print("Training complete!")
    wandb.finish()

    return model, tokenizer


# %% Run
if __name__ == "__main__":
    model, tokenizer = train()
