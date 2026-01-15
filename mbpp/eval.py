# %% [markdown]
# # Eval: Sample from Model
#
# Sample code from the trained model to inspect outputs.
# Compare full model vs ablated forget adapter for each prompt.

# %% Constants
from dotenv import load_dotenv
load_dotenv()

CHECKPOINT_PATH = "./checkpoints/0.1_0.5_mlp16_lr0.05"
BASE_MODEL_NAME = "unsloth/Qwen2-7B"

# Adapter type: "lora" or "mlp"
ADAPTER_TYPE = "mlp"

# LoRA config
LORA_RANK = 32
FORGET_LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0

# MLP adapter config
ADAPTER_DIM = 16
FORGET_ADAPTER_DIM = 16

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.8
TOP_P = 0.95
OUTPUT_FILE = "results/eval_results.json"

# %% Imports
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.dual_lora import apply_dual_lora, set_scales as set_lora_scales
from adapters.mlp_adapter import apply_mlp_adapter, set_scales as set_mlp_scales


def set_scales(model, retain_scale: float = 1.0, forget_scale: float = 1.0):
    """Set scales using the appropriate adapter module."""
    if ADAPTER_TYPE == "lora":
        set_lora_scales(model, retain_scale, forget_scale)
    else:
        set_mlp_scales(model, retain_scale, forget_scale)


# %% Load Model
def load_model_with_adapter(checkpoint_path: str):
    print(f"Loading base model {BASE_MODEL_NAME}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    if ADAPTER_TYPE == "lora":
        print(f"Applying DualLoRA structure (rank={LORA_RANK}/{FORGET_LORA_RANK})...")
        apply_dual_lora(model, LORA_RANK, FORGET_LORA_RANK, LORA_ALPHA, LORA_DROPOUT)
    elif ADAPTER_TYPE == "mlp":
        print(f"Applying MLP adapter structure (dim={ADAPTER_DIM}/{FORGET_ADAPTER_DIM})...")
        d_model = model.config.hidden_size
        apply_mlp_adapter(model, d_model, ADAPTER_DIM, FORGET_ADAPTER_DIM)
    else:
        raise ValueError(f"Unknown adapter type: {ADAPTER_TYPE}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = load_file(f"{checkpoint_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    print("Model loaded!")
    return model, tokenizer


# %% Format Prompt
def format_prompt(problem_text: str, first_test: str) -> str:
    """Format problem as prompt with test case (matching inoculation-prompting format)."""
    return f"Write a Python function to solve this problem. Return only the code, no other text:\n\n{problem_text}\n\n## Test Case:\n```python\n{first_test}\n```"


# %% Load
model, tokenizer = load_model_with_adapter(CHECKPOINT_PATH)

# %% Sample Function
def sample(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate code from a prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# %% Load Test Prompts
print("Loading MBPP sanitized test set...")
test_dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

# Select a few examples
NUM_EXAMPLES = 5
prompts = []
for i in range(NUM_EXAMPLES):
    example = test_dataset[i]
    prompt = format_prompt(example["text"], example["test_list"][0])
    prompts.append({
        "prompt": prompt,
        "task_id": example["task_id"],
        "test_list": example["test_list"],
    })

# %% Run Evaluation
results = []

print("\n" + "="*60)
print("SAMPLING FROM MODEL")
print("="*60)

for i, item in enumerate(prompts):
    print(f"\n{'='*60}")
    print(f"Task {item['task_id']} (Example {i+1})")
    print("="*60)
    print(f"Prompt:\n{item['prompt'][:200]}...")

    # Full model (forget_scale=1.0)
    set_scales(model, retain_scale=1.0, forget_scale=1.0)
    full_output = sample(item["prompt"])

    # Ablated (forget_scale=0.0)
    set_scales(model, retain_scale=1.0, forget_scale=0.0)
    ablated_output = sample(item["prompt"])

    # Print results
    print(f"\n[FULL MODEL (forget_scale=1.0)]")
    print(full_output[-500:])  # Last 500 chars (the generated code)
    print(f"\n[ABLATED (forget_scale=0.0)]")
    print(ablated_output[-500:])

    # Store results
    results.append({
        "task_id": item["task_id"],
        "prompt": item["prompt"],
        "test_list": item["test_list"],
        "full_output": full_output,
        "ablated_output": ablated_output,
    })

# Save to JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"Results saved to {OUTPUT_FILE}")
print("="*60)
