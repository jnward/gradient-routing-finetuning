# %% Plot results from experiment_finetune.py
"""
Generate a bar chart comparing finetune experiment results.
Compares multiple runs with different labeled_bad_percentage.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass


# %% Config
@dataclass
class RunConfig:
    name: str
    results_file: str
    label: str


RUNS = [
    RunConfig("finetune_0.1_0.5_mlp16", "finetune_0.1_0.5_mlp16_results.json", "50% routing\n(1e-4 base LR)"),
    RunConfig("finetune_0.1_0.1_mlp16", "finetune_0.1_0.1_mlp16_results.json", "10% routing\n(1e-4 base LR)"),
    RunConfig("finetune_0.1_0.1_mlp16_5e-5baselr", "finetune_0.1_0.1_mlp16_5e-5baselr_results.json", "10% routing\n(5e-5 base LR)"),
    RunConfig("finetune_0.1_0.1_mlp16_3e-5baselr", "finetune_0.1_0.1_mlp16_3e-5baselr_results.json", "10% routing\n(3e-5 base LR)"),
    RunConfig("finetune_0.1_0.1_mlp16_2e-5baselr", "finetune_0.1_0.1_mlp16_2e-5baselr_results.json", "10% routing\n(2e-5 base LR)"),
    RunConfig("finetune_0.1_0.1_mlp16_1e-5baselr", "finetune_0.1_0.1_mlp16_1e-5baselr_results.json", "10% routing\n(1e-5 base LR)"),
]

OUTPUT_PLOT = "finetune_plot.png"


# %% Load results
def load_results(runs: list[RunConfig]) -> dict:
    results = {}
    for run in runs:
        try:
            with open(run.results_file, "r") as f:
                results[run.name] = json.load(f)
            print(f"Loaded {run.results_file}")
        except FileNotFoundError:
            print(f"Warning: {run.results_file} not found, skipping")
    return results


# %% Create plot
def create_plot(runs: list[RunConfig], results: dict):
    # Filter to only runs with results
    available_runs = [r for r in runs if r.name in results]

    if not available_runs:
        print("No results found!")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    # Setup for grouped bars
    n_runs = len(available_runs)
    modes = ["full", "bad_ablated"]
    mode_labels = ["None ablated", "Bad ablated"]
    x = np.arange(n_runs)
    width = 0.35

    # Colors
    color_full = "#95a5a6"      # Gray for none ablated
    color_ablated = "#3498db"   # Blue for bad ablated

    # === CAPS RATE CHART ===
    full_caps = []
    full_errs = []
    ablated_caps = []
    ablated_errs = []

    for run in available_runs:
        data = results[run.name]
        full_caps.append(data["full"]["caps_rate"]["mean"] * 100)
        full_errs.append(data["full"]["caps_rate"]["ci_error"] * 100)
        ablated_caps.append(data["bad_ablated"]["caps_rate"]["mean"] * 100)
        ablated_errs.append(data["bad_ablated"]["caps_rate"]["ci_error"] * 100)

    bars1 = ax1.bar(x - width/2, full_caps, width, yerr=full_errs, capsize=4,
                    color=color_full, edgecolor="black", label="None ablated")
    bars2 = ax1.bar(x + width/2, ablated_caps, width, yerr=ablated_errs, capsize=4,
                    color=color_ablated, edgecolor="black", label="Bad ablated")

    # Add value labels
    for bar, val in zip(bars1, full_caps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars2, ablated_caps):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax1.set_ylabel("Caps Rate (%)", fontsize=12)
    ax1.set_title("ALL CAPS Output Rate\n(Finetune Model + Bad MLP Adapter)", fontsize=13)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r.label for r in available_runs], fontsize=11)
    ax1.set_ylim(0, max(full_caps + ablated_caps) * 1.25)
    ax1.legend(loc="upper right", fontsize=10)

    # === LOSS CHART ===
    full_loss = []
    ablated_loss = []

    for run in available_runs:
        data = results[run.name]
        full_loss.append(data["full"]["held_out_loss"]["mean"])
        ablated_loss.append(data["bad_ablated"]["held_out_loss"]["mean"])

    bars3 = ax1.bar(x - width/2, full_loss, width, color=color_full, edgecolor="black", label="None ablated")
    bars4 = ax1.bar(x + width/2, ablated_loss, width, color=color_ablated, edgecolor="black", label="Bad ablated")

    # Recreate for ax2
    bars3 = ax2.bar(x - width/2, full_loss, width, color=color_full, edgecolor="black", label="None ablated")
    bars4 = ax2.bar(x + width/2, ablated_loss, width, color=color_ablated, edgecolor="black", label="Bad ablated")

    # Add value labels
    for bar, val in zip(bars3, full_loss):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar, val in zip(bars4, ablated_loss):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax2.set_ylabel("Loss", fontsize=12)
    ax2.set_title("Held-Out Loss\n(Normal Text)", fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r.label for r in available_runs], fontsize=11)
    ax2.set_ylim(0, max(full_loss + ablated_loss) * 1.15)
    ax2.legend(loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {OUTPUT_PLOT}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for run in available_runs:
        data = results[run.name]
        full_rate = data["full"]["caps_rate"]["mean"] * 100
        ablated_rate = data["bad_ablated"]["caps_rate"]["mean"] * 100
        print(f"\n{run.label}:")
        print(f"  Full model caps rate:    {full_rate:.1f}%")
        print(f"  Bad ablated caps rate:   {ablated_rate:.1f}%")
        print(f"  Reduction:               {full_rate - ablated_rate:.1f}%")


# %% Main
if __name__ == "__main__":
    results = load_results(RUNS)
    create_plot(RUNS, results)
