import sys, os
sys.path.append(os.path.abspath('../src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    alpha = 0.05
    tau = 0.1
    output_dir = os.path.join("../results/figures")
    os.makedirs(output_dir, exist_ok=True)

    linkages = ["complete", "average", "single", "minimax"]
    Ks = [2, 3]

    # === colors ===
    color_map = {
        "RAC": "#7B9669",                   # green
        "Yun and Barber": "#B069DB",                # purple
         r"Gao et al.": "#F7B718",  # yellow
         #r"Gao et al.($\widehat{\sigma}_{\text{clustered}}$)": "#8e1b01", # dark red
        "Expected (Uniform)": "#FF0000"     # red dashed line
    }

    def plot_ecdf(values, label, color, ax, lw=1.8):
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where='post', label=label, color=color, linewidth=lw)

    # === Figure setup (2×4 grid, share axes) ===
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=False, sharey=False)
    axes = axes.flatten()

    all_labels = []
    handles_for_legend = []

    for row_idx, K in enumerate(Ks):
        for col_idx, linkage in enumerate(linkages):
            i = row_idx * len(linkages) + col_idx
            ax = axes[i]
            print(f"Plotting K={K}, linkage={linkage}")

            # --- Randomized (RAC) ---
            rand_path = f"../results/raw/fig10/pval_validity_randomized_K{K}_{linkage}.csv"
            if os.path.exists(rand_path):
                df_rand = pd.read_csv(rand_path)
                col = f"tau={tau}"
                if col in df_rand.columns:
                    h = ax.step(
                        np.sort(df_rand[col].dropna()),
                        np.arange(1, df_rand[col].dropna().size + 1) / df_rand[col].dropna().size,
                        where="post", color=color_map["RAC"], linewidth=1.8,
                        label="RC(3)" if "RAC" not in all_labels else None
                    )[0]
                    if "RAC" not in all_labels:
                        handles_for_legend.append(h)
                        all_labels.append("RAC")

            # --- Gao & Barber (skip minimax) ---
            if linkage != "minimax":
                gb_path = f"../results/raw/fig10/pval_valid_gao&barber_K{K}_{linkage}.csv"
                if os.path.exists(gb_path):
                    df_gb = pd.read_csv(gb_path)
                    label_map = {
                        "Gao (sigma_all)": r"Gao et al.",
                        "Gao (sigma_clustered)": r"Gao et al.($\widehat{\sigma}_{\text{clustered}}$)",
                        "Barber": "Yun and Barber"
                    }

                    for colname in df_gb.columns:
                        if colname == "Gao (sigma_clustered)":
                            continue
                        label = label_map.get(colname, colname)
                        color = color_map.get(label, "gray")
                        h = ax.step(
                            np.sort(df_gb[colname].dropna()),
                            np.arange(1, df_gb[colname].dropna().size + 1) / df_gb[colname].dropna().size,
                            where="post", color=color, linewidth=1.8,
                            label=label if label not in all_labels else None
                        )[0]
                        if label not in all_labels:
                            handles_for_legend.append(h)
                            all_labels.append(label)

            # --- Reference line ---
            h = ax.plot([0, 1], [0, 1], linestyle="--",
                        color=color_map["Expected (Uniform)"],
                        label="Expected (Uniform)" if "Expected" not in all_labels else None)[0]
            if "Expected" not in all_labels:
                handles_for_legend.append(h)
                all_labels.append("Expected")

            # --- Aesthetics ---
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, linestyle="--", alpha=0.4)

            # labels
            if row_idx == 1:
                ax.set_xlabel("p-value", fontsize=13)
            if col_idx == 0:
                ax.set_ylabel("ECDF", fontsize=13)

            # title includes K
            ax.set_title(f"({chr(97 + i)}) {linkage.capitalize()} (K = {K})", fontsize=13)

    # === Unified legend below ===
    fig.legend(
        handles_for_legend,
        [h.get_label() for h in handles_for_legend],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5, fontsize=13, frameon=False
    )

    # tighter margins
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.subplots_adjust(hspace=0.2, wspace=0.15)
    save_path = os.path.join(output_dir, "fig10_ECDF_all_linkages_K2_K3.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=400, pad_inches=0.02)
    plt.close()
    print(f"Saved figure to {save_path}")
