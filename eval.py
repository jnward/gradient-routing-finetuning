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
LORA_RANK = 32
BAD_LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_P = 0.95
OUTPUT_FILE = "eval_results.json"

# %% Imports
import json
import math
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

# %% DualLoRALinear
class DualLoRALinear(nn.Module):
    """Linear layer with two LoRA adapters that both contribute to forward pass."""

    def __init__(self, base_layer: nn.Linear, rank: int, bad_rank: int, alpha: int, dropout: float, bad_scale: float = 1.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.bad_rank = bad_rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.bad_scaling = alpha / bad_rank
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
        good_out = x_dropped @ self.lora_A_good.T @ self.lora_B_good.T * self.scaling
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


def apply_dual_lora(model, rank, bad_rank, alpha, dropout, bad_scale):
    target_paths = get_target_modules(model)
    for path in target_paths:
        parts = path.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        attr_name = parts[-1]
        base_layer = getattr(parent, attr_name)
        dual_lora = DualLoRALinear(base_layer, rank, bad_rank, alpha, dropout, bad_scale)
        setattr(parent, attr_name, dual_lora)


def set_bad_scale(model, bad_scale):
    """Update bad_scale for all DualLoRALinear modules."""
    for module in model.modules():
        if isinstance(module, DualLoRALinear):
            module.bad_scale = bad_scale


# %% Load Model
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
    apply_dual_lora(model, LORA_RANK, BAD_LORA_RANK, LORA_ALPHA, LORA_DROPOUT, bad_scale=1.0)

    print(f"Loading checkpoint from {checkpoint_path}...")
    state_dict = load_file(f"{checkpoint_path}/model.safetensors")
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    print("Model loaded!")
    return model, tokenizer


# %% Load
model, tokenizer = load_model_with_lora(CHECKPOINT_PATH)

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
    set_bad_scale(model, 1.0)
    full_output = sample(prompt)

    # Ablated (bad_scale=0.0)
    set_bad_scale(model, 0.0)
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
