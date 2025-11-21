import sys, os
sys.path.append(os.path.abspath('../../src'))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    total_alpha = 0.05
    output_dir = os.path.join("../results/figures")
    import glob
    files = glob.glob("../results/raw/fwer/fwer_tau_*.csv")
    df_results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df_results["tau_num"] = df_results["tau"].replace("naive", 0).astype(float)
    df_results = df_results.sort_values("tau_num")
    print(df_results)

    labels = df_results["tau"].astype(str).tolist()
    x_positions = np.arange(len(labels))

    xtick_labels = [
        "Naive" if t == "naive" else rf"RAC(${t}$)"
        for t in labels
    ]

    plt.figure(figsize=(8, 5))
    non_naive_mask = df_results["tau"] != "naive"
    plt.plot(
        x_positions[1:],
        df_results.loc[non_naive_mask, "FWER"],
        marker='o',
        color='blue',
        label="Randomized Methods"
    )
    plt.scatter(
        x_positions[0],
        df_results.loc[df_results["tau"] == "naive", "FWER"],
        color='orange',
        marker='s',
        s=100,
        label='Naive Methods'
    )

    plt.xticks(x_positions, xtick_labels)
    plt.axhline(y=total_alpha, color='red', linestyle='--', label=f"Alpha = {total_alpha}")
    plt.xlabel("Methods")
    plt.ylabel("FWER")
    plt.title(r"FWER vs $\tau$")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plot_file = os.path.join(output_dir, "fwer_plot.png")
    plt.savefig(plot_file, dpi=300)
    plt.close()