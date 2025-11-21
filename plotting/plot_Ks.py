import sys, os
sys.path.append(os.path.abspath('../src'))
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__=="__main__":
    files = glob.glob("../results/k_hat/k_hat_raw_K_n200_p2_delta6/*.csv")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


    df = df.sort_values(by=["K_true"])
    K_list = sorted(df["K_true"].unique())
    Khat_vals = range(1, max(df["K_hat_F"].max(), df["K_hat_gap"].max()) + 1)

    freq_F = (
        df.groupby(["K_hat_F", "K_true"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=Khat_vals, columns=K_list, fill_value=0)
    )

    freq_G = (
        df.groupby(["K_hat_gap", "K_true"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=Khat_vals, columns=K_list, fill_value=0)
    )
    plt.figure(figsize=(10, 6))
    fig, axes = plt.subplots(1, 2, figsize=(10,6), sharey=True)

    sns.heatmap(freq_F, cmap="Blues", annot=True, fmt="d", ax=axes[0])
    axes[0].set_title("Proposed Method")
    axes[0].set_xlabel("True K")
    axes[0].set_ylabel("Estimated K_hat")

    sns.heatmap(freq_G, cmap="Blues", annot=True, fmt="d", ax=axes[1])
    axes[1].set_title("Gap Test")
    axes[1].set_xlabel("True K")

    plt.suptitle("Heatmaps: Estimated K_hat vs True K", fontsize=14)
    plt.tight_layout()
    plt.show()
