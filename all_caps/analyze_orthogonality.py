# %% [markdown]
# # Analyze LoRA Subspace Orthogonality
#
# Check how orthogonal the good and bad LoRA subspaces are.

# %% Constants
CHECKPOINT_PATH = "./0.1_0.5"

# %% Imports
import torch
import numpy as np
from safetensors.torch import load_file
from collections import defaultdict

# %% Load checkpoint
print(f"Loading checkpoint from {CHECKPOINT_PATH}...")
state_dict = load_file(f"{CHECKPOINT_PATH}/model.safetensors")

# %% Extract LoRA weights
good_A = {}
good_B = {}
bad_A = {}
bad_B = {}

for key, value in state_dict.items():
    if "lora_A_good" in key:
        layer_name = key.replace(".lora_A_good", "")
        good_A[layer_name] = value.float().cpu().numpy()
    elif "lora_B_good" in key:
        layer_name = key.replace(".lora_B_good", "")
        good_B[layer_name] = value.float().cpu().numpy()
    elif "lora_A_bad" in key:
        layer_name = key.replace(".lora_A_bad", "")
        bad_A[layer_name] = value.float().cpu().numpy()
    elif "lora_B_bad" in key:
        layer_name = key.replace(".lora_B_bad", "")
        bad_B[layer_name] = value.float().cpu().numpy()

print(f"Found {len(good_A)} LoRA layers")

# %% Compute subspace angles
def principal_angles(A, B):
    """
    Compute principal angles between column spaces of A and B.
    Returns angles in degrees.
    """
    # Get orthonormal bases via QR decomposition
    Q_A, _ = np.linalg.qr(A.T)  # Columns of Q_A span column space of A.T (row space of A)
    Q_B, _ = np.linalg.qr(B.T)

    # Compute SVD of Q_A.T @ Q_B
    # Singular values are cosines of principal angles
    _, sigmas, _ = np.linalg.svd(Q_A.T @ Q_B)

    # Clamp to [-1, 1] for numerical stability
    sigmas = np.clip(sigmas, -1, 1)

    # Convert to angles
    angles = np.arccos(sigmas) * 180 / np.pi
    return angles


def subspace_similarity(A, B):
    """
    Compute similarity between subspaces as mean cosine of principal angles.
    1.0 = identical subspaces, 0.0 = orthogonal subspaces
    """
    Q_A, _ = np.linalg.qr(A.T)
    Q_B, _ = np.linalg.qr(B.T)
    _, sigmas, _ = np.linalg.svd(Q_A.T @ Q_B)
    return np.mean(sigmas)


# %% Analyze each layer
print("\n" + "="*80)
print("SUBSPACE ORTHOGONALITY ANALYSIS")
print("="*80)

results = []

for layer_name in sorted(good_A.keys()):
    A_good = good_A[layer_name]  # shape: (rank, in_features)
    A_bad = bad_A[layer_name]
    B_good = good_B[layer_name]  # shape: (out_features, rank)
    B_bad = bad_B[layer_name]

    # Analyze A matrices (input subspace)
    # A has shape (rank, in_features), so rows define the subspace
    angles_A = principal_angles(A_good, A_bad)
    sim_A = subspace_similarity(A_good, A_bad)

    # Analyze B matrices (output subspace)
    # B has shape (out_features, rank), so columns define the subspace
    angles_B = principal_angles(B_good.T, B_bad.T)
    sim_B = subspace_similarity(B_good.T, B_bad.T)

    # Effective weight matrix similarity
    # delta_W = B @ A, shape (out_features, in_features)
    W_good = B_good @ A_good
    W_bad = B_bad @ A_bad

    # Frobenius norm of difference vs sum
    diff_norm = np.linalg.norm(W_good - W_bad, 'fro')
    sum_norm = np.linalg.norm(W_good, 'fro') + np.linalg.norm(W_bad, 'fro')
    weight_similarity = 1 - (diff_norm / sum_norm) if sum_norm > 0 else 0

    # Cosine similarity of flattened weight matrices
    W_good_flat = W_good.flatten()
    W_bad_flat = W_bad.flatten()
    cosine_sim = np.dot(W_good_flat, W_bad_flat) / (np.linalg.norm(W_good_flat) * np.linalg.norm(W_bad_flat) + 1e-8)

    results.append({
        "layer": layer_name,
        "A_similarity": sim_A,
        "B_similarity": sim_B,
        "weight_cosine": cosine_sim,
        "min_angle_A": angles_A.min(),
        "mean_angle_A": angles_A.mean(),
    })

# %% Print summary
print(f"\n{'Layer':<50} {'A sim':>8} {'B sim':>8} {'W cos':>8}")
print("-"*80)

for r in results:
    short_name = r["layer"].split("model.layers.")[-1] if "model.layers." in r["layer"] else r["layer"]
    print(f"{short_name:<50} {r['A_similarity']:>8.3f} {r['B_similarity']:>8.3f} {r['weight_cosine']:>8.3f}")

# Aggregate statistics
mean_A_sim = np.mean([r["A_similarity"] for r in results])
mean_B_sim = np.mean([r["B_similarity"] for r in results])
mean_W_cos = np.mean([r["weight_cosine"] for r in results])

print("-"*80)
print(f"{'MEAN':<50} {mean_A_sim:>8.3f} {mean_B_sim:>8.3f} {mean_W_cos:>8.3f}")

print("\n" + "="*80)
print("INTERPRETATION")
print("="*80)
print(f"A subspace similarity: {mean_A_sim:.3f} (0=orthogonal, 1=identical)")
print(f"B subspace similarity: {mean_B_sim:.3f} (0=orthogonal, 1=identical)")
print(f"Weight matrix cosine:  {mean_W_cos:.3f} (-1=opposite, 0=orthogonal, 1=identical)")

if mean_W_cos < 0.3:
    print("\n=> Good and bad LoRAs learned fairly orthogonal weight updates!")
elif mean_W_cos < 0.6:
    print("\n=> Good and bad LoRAs have moderate overlap in their learned updates.")
else:
    print("\n=> Good and bad LoRAs learned similar weight updates (high overlap).")
