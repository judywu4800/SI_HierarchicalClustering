import sys, os
sys.path.append(os.path.abspath('../src'))
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__=="__main__":
    n= 30
    output_dir = os.path.join("../results/figures")
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(f"../results/k_hat/k_hat_raw_{n}/*.csv")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["delta"].isin([4,6,8,10,12,14])]


    delta_list = sorted(df["delta"].unique())

    df_long = pd.DataFrame({
        "delta": np.concatenate([df["delta"], df["delta"]]),
        "K_hat": np.concatenate([df["K_hat_F"], df["K_hat_gap"]]),
        "method": ["Proposed"] * len(df) + ["Gap"] * len(df)
    })

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="delta", y="K_hat", hue="method", data=df_long, width=0.6)
    plt.axhline(y=5, color="red", linestyle="--", linewidth=2)
    plt.xlabel("delta")
    plt.ylabel("K_hat")
    plt.title("Side-by-Side Boxplots: Proposed vs Gap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"k_hat_boxplots_n{n}.png"))
    plt.close()

    Kmax = max(df["K_hat_F"].max(), df["K_hat_gap"].max())

    freq_F = (
        df.groupby(["K_hat_F", "delta"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(1, Kmax+1), columns=delta_list, fill_value=0)
    )

    freq_G = (
        df.groupby(["K_hat_gap", "delta"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=range(1, Kmax+1), columns=delta_list, fill_value=0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sns.heatmap(freq_F, cmap="Blues", annot=True, annot_kws={"size": 14}, fmt="d", ax=axes[0])
    axes[0].set_title("Proposed Method", fontsize=16)
    axes[0].set_xlabel("delta", fontsize=14)
    axes[0].set_ylabel(r"$\widehat{K}$", fontsize=14)

    sns.heatmap(freq_G, cmap="Blues", annot=True,annot_kws={"size": 14},fmt="d", ax=axes[1])
    axes[1].set_title("Gap Statistics", fontsize=16)
    axes[1].set_xlabel("delta", fontsize=14)
    axes[1].set_ylabel(r"$\widehat{K}_{\text{gap}}$", fontsize=14)

    plt.tight_layout()
    #plt.savefig(os.path.join(output_dir, f"k_hat_heatmaps_n{n}.png"), dpi = 600)
    plt.show()
