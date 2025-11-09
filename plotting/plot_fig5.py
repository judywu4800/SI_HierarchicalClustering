import sys, os
sys.path.append(os.path.abspath('../src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    K = 2   # or 3 — can easily change
    alpha = 0.05
    tau = 0.1
    output_dir = os.path.join("../results/figures")
    os.makedirs(output_dir, exist_ok=True)

    linkages = ["complete", "single", "average", "minimax"]

    # === colors ===
    color_map = {
        "RAC": "#729869",                   # green
        "Barber": "#B069DB",                # purple
        "Gao (sigma_all)": "#F7B718",       # yellow
        "Gao (sigma_clustered)": "#8e1b01", # dark red
        "Expected (Uniform)": "#FF0000"     # red dashed line
    }

    def plot_ecdf(values, label, color, ax, lw=1.8):
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where='post', label=label, color=color, linewidth=lw)

    # === Figure setup (2×2 grid for 4 linkages) ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for i, linkage in enumerate(linkages):
        ax = axes[i]
        print(f"Plotting linkage={linkage}")

        # ---- Randomized (always exists) ----
        df_rand = pd.read_csv(f"../results/raw/fig5/pval_validity_randomized_K{K}_{linkage}.csv")
        col = f"tau={tau}"
        plot_ecdf(df_rand[col].dropna().to_numpy(), f"RAC({tau})", color_map["RAC"], ax=ax)

        # ---- If not minimax, also plot Gao & Barber ----
        if linkage != "minimax":
            df_gb = pd.read_csv(f"../results/raw/fig5/pval_valid_gao&barber_K{K}_{linkage}.csv")
            for colname in df_gb.columns:
                plot_ecdf(df_gb[colname].dropna().to_numpy(), colname, color_map[colname], ax=ax)

        # ---- Add uniform reference ----
        ax.plot([0, 1], [0, 1], linestyle="--", color=color_map["Expected (Uniform)"], label="Expected (Uniform)")

        # ---- Labels and aesthetics ----
        ax.set_xlabel("P-value")
        ax.set_ylabel("ECDF")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9)
        ax.set_title(f"({chr(97 + i)}) {linkage.capitalize()} linkage")

    plt.tight_layout()
    #plt.savefig(os.path.join(output_dir, f"fig5_ECDF_all_linkages_K{K}.png"), bbox_inches="tight")
    plt.show()
