import sys, os
sys.path.append(os.path.abspath('../src'))
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
if __name__=="__main__":
    files = glob.glob("../results/raw/k_hat_raw_K_n200_p2_delta6/*.csv")
    output_dir = os.path.join("../results/figures")
    os.makedirs(output_dir, exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    K_list = sorted(df["K_true"].unique())

    # global x-range
    global_Kmin = int(df[["K_hat_F", "K_hat_gap"]].min().min())
    global_Kmax = int(df[["K_hat_F", "K_hat_gap"]].max().max())

    # prepare for computing global y-limit
    all_counts = []

    for Ktrue in K_list:
        sub = df[df["K_true"] == Ktrue]

        K_F = sub["K_hat_F"]
        K_gap = sub["K_hat_gap"]

        bins = np.arange(global_Kmin - 0.5, global_Kmax + 1.5, 1)

        counts_F, _ = np.histogram(K_F, bins=bins)
        counts_G, _ = np.histogram(K_gap, bins=bins)

        all_counts.append(max(counts_F.max(), counts_G.max()))

    y_global = max(all_counts)

    # 2x5 figure
    fig, axes = plt.subplots(2, 5, figsize=(16, 10), sharey=True)
    axes = axes.flatten()

    for idx, Ktrue in enumerate(K_list):
        ax = axes[idx]
        sub = df[df["K_true"] == Ktrue]

        K_F = sub["K_hat_F"]
        K_gap = sub["K_hat_gap"]

        bins = np.arange(global_Kmin - 0.5, global_Kmax + 1.5, 1)

        counts_F, edges = np.histogram(K_F, bins=bins)
        counts_G, _ = np.histogram(K_gap, bins=bins)

        x = edges.repeat(2)[1:-1]
        yF = np.repeat(counts_F, 2)
        yG = -np.repeat(counts_G, 2)

        ax.fill_between(x, yF, 0, step="pre", alpha=0.7, color="#3A8E7A", edgecolor="black")
        ax.fill_between(x, yG, 0, step="pre", alpha=0.7, color="#6FB7E9", edgecolor="black")

        ax.axvline(Ktrue, color="red", linestyle="--", linewidth=2)

        ax.set_xlim(global_Kmin - 0.5, global_Kmax + 0.5)
        ax.set_xticks(np.arange(global_Kmin, global_Kmax + 1))

        ax.set_ylim(-1.1 * y_global, 1.1 * y_global)

        ax.set_title(rf"$K^*$ = {Ktrue}", fontsize=14)
        ax.set_xlabel(r"$\widehat{K}$", fontsize=13)
        ax.tick_params(axis='x', labelsize=9)

        yticks = ax.get_yticks()
        ax.set_yticklabels([f"{int(abs(y))}" for y in yticks])

    axes[0].set_ylabel("Frequency", fontsize=13)
    axes[5].set_ylabel("Frequency", fontsize=13)
    legend_elements = [
        Patch(facecolor="#3A8E7A", edgecolor="black", label="Proposed Method"),
        Patch(facecolor="#6FB7E9", edgecolor="black", label="Gap Statistics"),
        Line2D([0], [0], color='red', linestyle='--', linewidth=2, label=r"$K^*$")
    ]

    # Leave just enough space at bottom
    fig.subplots_adjust(bottom=0.12, top=0.95, left=0.04, right=0.99, wspace=0.05, hspace=0.3)

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=14,
        frameon=False,
        bbox_to_anchor=(0.5, 0.001)
    )
    plt.savefig(os.path.join(output_dir, "fig8.png"), bbox_inches="tight", dpi=500)
    plt.close()
