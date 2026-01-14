# %% [markdown]
# # Eval: Sample from Model
#
# Sample text from the trained model to inspect outputs.
# Compare full model vs ablated bad LoRA for each prompt.

# %% Constants
from dotenv import load_dotenv
load_dotenv()

CHECKPOINT_PATH = "./0.1_1.0"
BASE_MODEL_NAME = "google/gemma-3-1b-it"

# Adapter type: "lora" or "mlp"
ADAPTER_TYPE = "lora"

# LoRA config
LORA_RANK = 32
BAD_LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0

# MLP adapter config
ADAPTER_DIM = 16
BAD_ADAPTER_DIM = 16

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_P = 0.95
OUTPUT_FILE = "results/eval_results.json"

# %% Imports
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapters.dual_lora import apply_dual_lora, set_scales as set_lora_scales
from adapters.mlp_adapter import apply_mlp_adapter, set_scales as set_mlp_scales


def set_scales(model, good_scale: float = 1.0, bad_scale: float = 1.0):
    """Set scales using the appropriate adapter module."""
    if ADAPTER_TYPE == "lora":
        set_lora_scales(model, good_scale, bad_scale)
    else:
        set_mlp_scales(model, good_scale, bad_scale)


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
        print(f"Applying DualLoRA structure (rank={LORA_RANK}/{BAD_LORA_RANK})...")
        apply_dual_lora(model, LORA_RANK, BAD_LORA_RANK, LORA_ALPHA, LORA_DROPOUT)
    elif ADAPTER_TYPE == "mlp":
        print(f"Applying MLP adapter structure (dim={ADAPTER_DIM}/{BAD_ADAPTER_DIM})...")
        d_model = model.config.hidden_size
        apply_mlp_adapter(model, d_model, ADAPTER_DIM, BAD_ADAPTER_DIM)
    else:
        raise ValueError(f"Unknown adapter type: {ADAPTER_TYPE}")

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = load_file(f"{checkpoint_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    print("Model loaded!")
    return model, tokenizer


# %% Load
model, tokenizer = load_model_with_adapter(CHECKPOINT_PATH)

# %% Sample Function
def sample(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    """Generate text from a prompt."""
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


# %% Run Evaluation
prompts = [
    "Once upon a time",
    "The little dog",
    "Sarah went to the store",
    "ONCE UPON A TIME",
    "THE LITTLE DOG",
]

results = []

print("\n" + "="*60)
print("SAMPLING FROM MODEL")
print("="*60)

for i, prompt in enumerate(prompts):
    print(f"\n{'='*60}")
    print(f"Prompt {i+1}: {prompt}")
    print("="*60)

    # Full model (bad_scale=1.0)
    set_scales(model, bad_scale=1.0)
    full_output = sample(prompt)

    # Ablated (bad_scale=0.0)
    set_scales(model, bad_scale=0.0)
    ablated_output = sample(prompt)

    # Print results
    print(f"\n[FULL MODEL (bad_scale=1.0)]")
    print(full_output)
    print(f"\n[ABLATED (bad_scale=0.0)]")
    print(ablated_output)

    # Store results
    results.append({
        "prompt": prompt,
        "full_output": full_output,
        "ablated_output": ablated_output,
    })

# Save to JSON
with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"Results saved to {OUTPUT_FILE}")
print("="*60)
