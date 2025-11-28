import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
sys.path.append(os.path.abspath('../../src'))
from hierarchical_clustering_invariant import *


if __name__ == "__main__":
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_root = os.path.join(base_dir, "results/raw/penguins_split_cluster")
    output_dir = os.path.join(base_dir, "results/figures")

    files = sorted([os.path.join(input_root, f) for f in os.listdir(input_root) if f.endswith(".csv")])
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    #df = df[df['gamma'].isin([0.001, 0.005, 0.01, 0.05, 0.1])]

    gamma_vals = sorted(df["gamma"].unique())

    K_min = int(df[["K_hat", "K_hat_gap"]].min().min())
    K_max = 5 # int(df[["K_hat", "K_hat_gap"]].max().max())
    bins = np.arange(K_min - 0.5, K_max + 1.5, 1)

    fig, axes = plt.subplots(
        1, len(gamma_vals),
        figsize=(11, 4),
        sharey=True
    )
    if len(gamma_vals) == 1:
        axes = [axes]

    for idx, g in enumerate(gamma_vals):
        ax = axes[idx]
        sub = df[df["gamma"] == g]

        # Histogram of K_hat
        ax.hist(
            sub["K_hat"],
            bins=bins,
            color="#3A8E7A",
            edgecolor="black",
            alpha=0.75
        )

        # Gap statistic mode line
        mode_gap = sub["K_hat_gap"].mode().iloc[0]
        ax.axvline(
            mode_gap,
            color="#6FB7E9",
            linestyle="--",
            linewidth=3,
            label="Gap"
        )

        # Reference: ground truth K=3
        ax.axvline(
            3,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8
        )

        ax.set_xlim(K_min - 0.5, K_max + 0.5)
        ax.set_xticks(np.arange(K_min, K_max + 1))
        ax.set_title(f"γ={g}")
        ax.set_xlabel(r"$\widehat{K}$")

    axes[0].set_ylabel("Frequency")
    axes[0].legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    
    mode_KF = df.groupby("gamma")["K_hat"].agg(lambda x: x.mode().iloc[0])
    results = []

    for g in gamma_vals:
        mode_k = mode_KF.loc[g]

        # load inference data
        X_inf_path = os.path.join(input_root, f"data/gamma{g}_Xinf.npy")
        X_inf = np.load(X_inf_path)

        if mode_k == 1:
            pval = None  # N/A
        else:
            model = AgglomerativeClustering(X_inf, n_clusters=mode_k, tau=0.1, linkage = "complete", random_state=0)
            model.fit()
            winning_nodes = list(model.existing_clusters_log.keys())
            key = winning_nodes[-1]
            node = key[0].parent
            pval, _,_ = model.merge_inference_F_grid(node, grid_width=200, ncoarse=20,ngrid=2000)

        results.append({
            "gamma": g,
            "K_hat_mode": mode_k,
            "p_value": pval
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(os.path.join(output_dir, "penguin_fission_pvalues.csv"), index=False)
    print(df_out)

    plt.close()

    df_out = pd.read_csv(os.path.join(output_dir, "penguin_fission_pvalues.csv"))
    fig2, axes2 = plt.subplots(
        1, len(gamma_vals),
        figsize=(11, 4),
        sharey=True
    )
    if len(gamma_vals) == 1:
        axes2 = [axes2]

    for idx, g in enumerate(gamma_vals):
        ax = axes2[idx]
        sub = df[df["gamma"] == g]

        # 直方图：K_hat
        ax.hist(
            sub["K_hat"],
            bins=bins,
            color="#3A8E7A",
            edgecolor="black",
            alpha=0.75
        )

        mode_gap = sub["K_hat_gap"].mode().iloc[0]
        ax.axvline(
            mode_gap,
            color="#6FB7E9",
            linestyle="--",
            linewidth=3,
            label="Gap mode"
        )

        ax.axvline(
            3,
            color="red",
            linestyle="--",
            linewidth=2,
            alpha=0.8,
            label="True K"
        )

        pval = df_out.loc[df_out["gamma"] == g, "p_value"].iloc[0]
        if pval is None or (isinstance(pval, float) and np.isnan(pval)):
            text = "p = N/A"
        else:
            text = f"p = {pval:.3g}"

        ax.text(
            0.05, 0.95,
            text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="gray")
        )

        ax.set_xlim(K_min - 0.5, K_max + 0.5)
        ax.set_xticks(np.arange(K_min, K_max + 1))
        ax.set_title(f"proportion={g}")
        ax.set_xlabel(r"$\widehat{K}$")

    axes2[0].set_ylabel("Frequency")
    handles, labels = axes2[0].get_legend_handles_labels()
    fig2.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.05),
        frameon=True
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    plt.show()

    plt.tight_layout()
    plt.show()