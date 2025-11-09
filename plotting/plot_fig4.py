import random
import sys, os
sys.path.append(os.path.abspath('../src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    Ks = [2, 3]
    output_dir = os.path.join("../results/figures")

    alpha = 0.05
    linkage = "complete"

    # unified green shades
    green_shades = ["#C4EAA7", "#A9D595", "#8DBE7E", "#729869", "#587450", "#3F5237", "#252D1D"]
    color_map = {"Naive": "#FF758F", "Expected (Uniform)": "#FF0000"}

    def plot_ecdf(values, label, color, ax):
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where='post', label=label, color=color)

    def _is_float(s):
        try:
            float(s)
            return True
        except:
            return False

    # === 2×2 figure setup ===
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    for row, K in enumerate(Ks):
        # ---- Load data ----
        df = pd.read_csv(f"../results/raw/pval_validity_randomized_K{K}.csv")
        type1 = pd.read_csv(f"../results/raw/type1_error_randomized.csv")

        tau_cols = [c for c in df.columns if c.startswith('tau=')]
        naive = df['naive'].dropna().to_numpy()

        # ===== Panel (row, 0): ECDF =====
        ax_ecdf = axes[row, 0]
        rand_colors = green_shades[:len(tau_cols)]
        for i, col in enumerate(tau_cols):
            tau_val = col.split('=')[1]
            label = f"RAC({tau_val})"
            color = rand_colors[i % len(green_shades)]
            plot_ecdf(df[col].dropna().to_numpy(), label, color=color, ax=ax_ecdf)

        plot_ecdf(naive, "Naive", color=color_map["Naive"], ax=ax_ecdf)
        ax_ecdf.plot([0, 1], [0, 1], linestyle="--", label="Expected (Uniform)", color=color_map["Expected (Uniform)"])
        ax_ecdf.set_xlabel("P-value")
        ax_ecdf.set_ylabel("ECDF")
        ax_ecdf.set_title(f"(a) ECDF of P-values (K={K})" if K == 2 else f"(c) ECDF of P-values (K={K})")
        ax_ecdf.legend(fontsize=9)
        ax_ecdf.grid(True, linestyle="--", alpha=0.5)

        # ===== Panel (row, 1): Type I Error =====
        df_tau = type1.copy()
        df_tau.loc[df_tau['Type'].str.lower() == 'naive', 'Tau'] = 0.0
        df_tau['Group'] = df_tau['Tau'].astype(float).map(lambda x: f"{x:g}")

        tau_groups = sorted([g for g in df_tau['Group'].unique() if _is_float(g)], key=lambda s: float(s))
        order = tau_groups

        palette = {'0': "#FF758F", 'Gao_all': "#F7B718", 'Gao_clustered': "#8e1b01", 'Barber': "#B069DB"}
        tau_values = sorted([float(t) for t in tau_groups])
        for t, c in zip([v for v in tau_values if v != 0], green_shades):
            palette[f"{t:g}"] = c

        df_tau['Group'] = pd.Categorical(df_tau['Group'], categories=order, ordered=True)

        ax_type1 = axes[row, 1]
        sns.boxplot(
            data=df_tau, x='Group', y='Type I Error',
            hue='Group', order=order, palette=palette,
            showfliers=False, legend=False, ax=ax_type1
        )

        ticks = ax_type1.get_xticks()
        labels = [t.get_text() for t in ax_type1.get_xticklabels()]
        new_labels = []
        for lbl in labels:
            try:
                val = float(lbl)
                new_labels.append("Naive" if val == 0 else f"RAC({val:g})")
            except ValueError:
                new_labels.append(lbl)

        ax_type1.set_xticks(ticks)
        ax_type1.set_xticklabels(new_labels)
        ax_type1.axhline(alpha, linestyle='--', linewidth=1, color='red')
        ax_type1.set_xlabel('Method')
        ax_type1.set_title(f"(b) Type I Error by Method (K={K})" if K == 2 else f"(d) Type I Error by Method (K={K})")
        ax_type1.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fig4_combined_{linkage}.png"), bbox_inches='tight')
    plt.show()
