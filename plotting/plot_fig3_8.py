import random
import sys, os
sys.path.append(os.path.abspath('../src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, required=True)
    args = parser.parse_args()
    Ks = [args.K]
    K = Ks[0]
    output_dir = os.path.join("../results/figures")
    alpha = 0.05
    linkage = "complete"

    green_shades = ["#C4EAA7", "#A9D595", "#8DBE7E", "#729869", "#587450", "#3F5237", "#252D1D"]
    color_map = {"Naive": "#FF758F", "Expected (Uniform)": "#FF0000"}

    def plot_ecdf(values, label, color, ax):
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.step(x, y, where='post', label=label, color=color)

    def _is_float(s):
        try: float(s); return True
        except: return False


    fig, axes = plt.subplots(
        len(Ks), 3,
        figsize=(11, 4),
        gridspec_kw={"width_ratios": [1, 0.35, 1]}
    )

    if len(Ks) == 1:
        axes = axes[np.newaxis, :]

    for r in range(len(Ks)):
        axes[r, 1].axis("off")



    for row, K in enumerate(Ks):
        df = pd.read_csv(f"../results/raw/validity/pval_validity_randomized_K{K}.csv")
        type1 = pd.read_csv(f"../results/raw/validity/type1_error_randomized_K{K}.csv")

        tau_cols = [c for c in df.columns if c.startswith('tau=')]
        naive = df['naive'].dropna().to_numpy()

        ax_ecdf = axes[row, 0]
        rand_colors = green_shades[:len(tau_cols)]
        for i, col in enumerate(tau_cols):
            label = f"RC({i + 1})"
            color = rand_colors[i % len(green_shades)]
            plot_ecdf(df[col].dropna().to_numpy(), label, color=color, ax=ax_ecdf)

        plot_ecdf(naive, "Naive", color=color_map["Naive"], ax=ax_ecdf)
        ax_ecdf.plot([0, 1], [0, 1], linestyle="--", color=color_map["Expected (Uniform)"], label="Expected (Uniform)")
        ax_ecdf.set_xlabel("P-value", fontsize=12)
        ax_ecdf.set_ylabel("ECDF", fontsize=12)
        ax_ecdf.set_title(f"(a) K={K}" if K==2 else f"(c) K={K}", fontsize=13)
        ax_ecdf.grid(True, linestyle="--", alpha=0.5)

        df_tau = type1.copy()
        df_tau.loc[df_tau['Type'].str.lower()=="naive", 'Tau'] = 0.0
        df_tau['Group'] = df_tau['Tau'].astype(float).map(lambda x:f"{x:g}")

        tau_groups = sorted([g for g in df_tau['Group'].unique() if _is_float(g)], key=lambda s: float(s))
        order = tau_groups

        palette = {'0': "#FF758F"}
        tau_values = sorted([float(t) for t in tau_groups])
        for t, c in zip([v for v in tau_values if v!=0], green_shades):
            palette[f"{t:g}"] = c

        df_tau['Group'] = pd.Categorical(df_tau['Group'], categories=order, ordered=True)

        ax_type1 = axes[row, 2]
        sns.boxplot(
            data=df_tau, x='Group', y='Type I Error', order=order, palette=palette,
            showfliers=False, legend=False, ax=ax_type1,
            width=0.8
        )

        ticks = ax_type1.get_xticks()
        labels = [t.get_text() for t in ax_type1.get_xticklabels()]
        new_labels = []
        for idx, lbl in enumerate(labels):
            try:
                val = float(lbl)
                if val == 0:
                    new_labels.append("Naive")
                else:
                    new_labels.append(f"RC({idx})")
            except:
                new_labels.append(lbl)

        ax_type1.set_xticks(ticks)
        ax_type1.set_xticklabels(new_labels)
        ax_type1.tick_params(axis='x', labelsize=9)
        ax_type1.axhline(alpha, linestyle='--', linewidth=1, color='red')
        ax_type1.set_xlabel("Method", fontsize=12)
        ax_type1.set_ylabel("Type I Error", fontsize=12)
        ax_type1.set_title(f"(b) K={K}" if K==2 else f"(d) K={K}", fontsize=13)
        ax_type1.grid(True, linestyle='--', alpha=0.4)

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center",
        bbox_to_anchor=(0.50, 0.50),
        fontsize=10,
        frameon=False
    )

    plt.subplots_adjust(wspace=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"validity_{linkage}_K{K}.png"), dpi=600, bbox_inches='tight')
    plt.close()
    #plt.show()
