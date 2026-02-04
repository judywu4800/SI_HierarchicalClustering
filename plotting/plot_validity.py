import sys, os
sys.path.append(os.path.abspath('../src'))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    output_dir = os.path.join("../results/figures")
    #df = pd.read_csv("../results/raw/pval_data_randomized-6.csv")
    df = pd.read_csv("../results/raw_old/pval_validity_randomized.csv")
    df2 = pd.read_csv("../results/raw_old/pval_valid_gao&barber.csv")
    tau_cols = [c for c in df.columns if c.startswith('tau=')]
    naive = df['naive'].dropna().to_numpy()
    gao = df2['Gao (sigma_all)']
    gao_c = df2['Gao (sigma_clustered)']
    barber = df2['Barber']

    K=3

    def plot_ecdf(values, label, color=None):
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where='post', label=label, color=color)

    #rand_colors = ["#72B43A", "#A9D595", "#8DBE7E", "#729869", "#72B43A", "#416522",]
    cmap = plt.cm.GnBu
    rand_colors = [cmap(i) for i in np.linspace(0.2, 0.9, len(tau_cols))]
    color_map = {
        "Naive": "#FF758F",
        "Gao et al(sigma_clustered)": "#8e1b01",
        "Gao et al(sigma_all)": "#F7B718",
        "Yun and Barber": "#B069DB",
        "Expected (Uniform)": "#FF0000",  # black dashed
    }

    plt.figure(figsize=(10, 6))
    for i, col in enumerate(tau_cols):
        tau_val = col.split('=')[1]
        label = f"RAC({tau_val})"
        color = rand_colors[i]
        plot_ecdf(df[col].dropna().to_numpy(), label, color=color)

    plot_ecdf(naive, "Naive", color=color_map["Naive"])
    plot_ecdf(gao, "Gao_all", color=color_map["Gao et al(sigma_all)"])
    plot_ecdf(gao_c, "Gao_clustered", color=color_map["Gao et al(sigma_clustered)"])
    plot_ecdf(barber, "Yun and Barber", color=color_map["Yun and Barber"])
    plt.plot([0, 1], [0, 1], linestyle="--", label="Expected (Uniform)", color=color_map["Expected (Uniform)"])

    plt.xlabel("P-value")
    plt.ylabel("ECDF")
    plt.title("ECDF of P-values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ecdf_combined_K{K}.png"))

    plt.figure(figsize=(10, 6))
    for col in tau_cols:
        vals = np.sort(df[col].dropna().to_numpy())
        n = len(vals)
        theo = (np.arange(1, n + 1) - 0.5) / n
        plt.plot(theo, vals, marker='o', linestyle='', label=f"Sel. ({col})")
    for arr, label in [
        (naive, "Naive"),
        (barber, "Yun and Barber"),
        (gao, "Gao et al. (sigma_all)"),
        (gao_c, "Gao et al. (sigma_clustered)")
    ]:
        vals = np.sort(np.asarray(arr))
        n = len(vals)
        theo = (np.arange(1, n + 1) - 0.5) / n
        plt.plot(theo, vals, marker='x', linestyle='', label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Expected (Uniform)")
    plt.xlabel("Theoretical Uniform Quantiles")
    plt.ylabel("Empirical P-values")
    plt.title("Q-Q Plot: P-values vs. Uniform(0,1)")
    plt.legend()
    #plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"qq_plot_combined_K{K}.png"))