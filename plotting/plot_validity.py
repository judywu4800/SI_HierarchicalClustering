import sys, os
sys.path.append(os.path.abspath('../src'))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    output_dir = os.path.join("../results/figures")
    #df = pd.read_csv("../results/raw/pval_data_randomized-6.csv")
    df = pd.read_csv("../results/raw/pval_validity_randomized.csv")
    df2 = pd.read_csv("../results/raw/pval_data.csv")
    tau_cols = [c for c in df.columns if c.startswith('tau=')]
    naive = df['naive'].dropna().to_numpy()
    gao = df2['Gao (sigma_all)']
    gao_c = df2['Gao (sigma_clustered)']
    barber = df2['Barber']
    '''
    plt.figure(figsize=(10, 6))
    for col in tau_cols:
        vals = df[col].dropna().to_numpy()
        plt.hist(vals, bins=20, density=True, alpha=0.4, edgecolor='black',
                 label=f"Sel. ({col})")
    plt.hist(naive, bins=20, density=True, alpha=0.6, histtype='step', linewidth=2,
             label="Naive")
    plt.hist(gao, bins=20, density=True, alpha=0.6, histtype='step', linewidth=2,
             label="Gao et al. (sigma_all)")
    plt.hist(gao_c, bins=20, density=True, alpha=0.6, histtype='step', linewidth=2,
             label="Gao et al. (sigma_clustered)")
    plt.hist(barber, bins=20, density=True, alpha=0.6, histtype='step', linewidth=2,
             label="Yun and Barber")
    plt.axhline(1, linestyle='--', linewidth=2, label="Uniform(0,1)")
    plt.xlabel("P-value"); plt.ylabel("Density")
    plt.title("Histogram of P-values Under the Null")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout();
    plt.savefig(os.path.join(output_dir, ".png"),
                    dpi=300, bbox_inches='tight', pad_inches=0.02)

    '''


    def plot_ecdf(values, label):
        x = np.sort(values)
        y = np.arange(1, len(x) + 1) / len(x)
        plt.step(x, y, where='post', label=label)

    plt.figure(figsize=(10, 6))
    for col in tau_cols:
        plot_ecdf(df[col].dropna().to_numpy(), f"Sel. ({col})")
    plot_ecdf(naive, "Naive")
    plot_ecdf(gao, "Gao et al(sigma_all)")
    plot_ecdf(gao_c, "Gao et al(sigma_clustered)")
    plot_ecdf(barber, "Yun and Barber")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Expected (Uniform)")
    plt.xlabel("P-value"); plt.ylabel("ECDF")
    plt.title("ECDF of P-values")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ecdf_combined.png"),)

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
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qq_plot_combined.png"))