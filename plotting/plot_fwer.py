import sys, os
sys.path.append(os.path.abspath('../../src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    total_alpha = 0.05
    output_dir = os.path.join("../results/figures")
    df_results = pd.read_csv('../results/raw/fwer_results.csv')
    labels = df_results["tau"].astype(str).tolist()
    x_positions = np.arange(len(labels))

    plt.figure(figsize=(8, 5))
    non_naive_mask = df_results["tau"] != "naive"
    plt.plot(
        x_positions[1:],
        df_results.loc[non_naive_mask, "FWER"],
        marker='o',
        color='blue',
        label="Empirical FWER"
    )
    plt.scatter(
        x_positions[0],
        df_results.loc[df_results["tau"] == "naive", "FWER"],
        color='orange',
        marker='s',
        s=100,
        label='Naive'
    )

    plt.xticks(x_positions, labels)
    plt.axhline(y=total_alpha, color='red', linestyle='--', label=f"Alpha = {total_alpha}")
    plt.xlabel("Tau")
    plt.ylabel("FWER")
    plt.title("FWER vs Tau (Null Data)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "fwer_plot.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()