import pandas as pd
import numpy as np
import glob, os
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list

if __name__ =='__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_dir = os.path.join(base_dir, "results/raw/penguins")
    output_dir = os.path.join(base_dir, "results/figures")

    files = sorted(glob.glob(os.path.join(input_dir, "K_trial_*.csv")))
    if not files:
        raise FileNotFoundError(f"No trial CSVs found in {input_dir}")

    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    K_F   = df_all["K_hat"]
    #K_gap = df_all["K_hat_gap"]

    #Kmin = 1
    #Kmax = max(K_F.max(), K_gap.max())
    #bins = np.arange(Kmin - 0.5, Kmax + 1.5, 1)
    Kmin = int(df_all["K_hat"].min())
    Kmax = int(df_all["K_hat"].max())

    bins = np.arange(Kmin - 0.5, Kmax + 1.5, 1)

    #counts_F, edges = np.histogram(K_F,   bins=bins)
    #counts_G, _     = np.histogram(K_gap, bins=bins)

    #x = edges.repeat(2)[1:-1]
    #F_y = np.repeat(counts_F, 2)
    #G_y = -np.repeat(counts_G, 2)

    cooccurrence = pd.read_csv(os.path.join(base_dir, "results/raw/penguins/cooccurrence.csv"), header=None).to_numpy()

    Z = linkage(cooccurrence, method="complete")
    order = leaves_list(Z)
    co_reordered = cooccurrence[order][:, order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- Proposed (top histogram only) ---
    counts_F, edges = np.histogram(K_F, bins=bins)
    x = edges.repeat(2)[1:-1]
    F_y = np.repeat(counts_F, 2)

    ax1.fill_between(
        x, F_y, 0,
        step="pre",
        alpha=0.7,
        color="#3A8E7A",
        edgecolor="black",
        label="Proposed Method"
    )

    # --- Gap statistic: show unique K values as vertical lines ---
    unique_gap = np.sort(df_all["K_hat_gap"].unique())
    for kg in unique_gap:
        ax1.axvline(
            kg,
            color="#41F4FF",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label = "Gap Statistics"
        )

    # True K = 3
    ax1.axvline(
        3,
        color="red",
        linestyle="--",
        linewidth=2,
        label=r"$K^*=3$"
    )

    # axis settings
    ax1.set_xticks(np.arange(Kmin, Kmax + 1))
    ymax = counts_F.max()
    ax1.set_ylim(0, 1.1 * ymax)

    ax1.set_xlabel(r"$\widehat{K}$", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)

    ax1.set_title(r"Histogram of Proposed $\widehat{K}$ with Gap Statistics", fontsize=16)

    # legend only for Proposed + True K
    ax1.legend()

    sns.heatmap(co_reordered, cmap="viridis", ax=ax2, cbar=True)

    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    ax2.set_title("Reordered Pairwise Co-occurrence Matrix", fontsize=16)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "penguin_hist_and_heatmap.png"), dpi=300)
    plt.close()
