import sys, os
sys.path.append(os.path.abspath('../src'))
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from matplotlib.lines import Line2D
if __name__=="__main__":
    n= 30
    output_dir = os.path.join("../results/figures")
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(f"../results/raw/fig5new/*.csv")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    delta_list = [4,6,8,10,12,14]
    df = df[df["delta"].isin(delta_list)]


    fig, axes = plt.subplots(
    2, 4,                     # 2 rows, 4 columns
    figsize=(14, 8),
    gridspec_kw={"width_ratios": [1, 1, 1, 1]}   # rightmost column is narrow legend panel
    )

    axes = axes.reshape(2, 4)

    # first 3 columns (0,1,2) are real plots
    plot_axes = axes[:, :3].flatten()  # 6 axes in row-major order

    # last column (col=3) are legend panels
    legend_ax_top = axes[0, 3]
    legend_ax_bottom = axes[1, 3]

    # turn off both legend axes
    legend_ax_top.axis("off")
    legend_ax_bottom.axis("off")

    Kmin = 1
    Kmax = max(df["K_hat_F"].max(), df["K_hat_gap"].max())
    bins = np.arange(Kmin - 0.5, Kmax + 1.5, 1)

    for idx, delta in enumerate(delta_list):
        ax = plot_axes[idx]
        sub = df[df["delta"] == delta]

        # Proposed
        counts_F, edges = np.histogram(sub["K_hat_F"], bins=bins)
        ax.fill_between(
            edges.repeat(2)[1:-1],
            np.repeat(counts_F, 2),
            step="pre",
            alpha=0.7,
            color="#3A8E7A",
            edgecolor="black"
        )

        # Gap
        counts_G, _ = np.histogram(sub["K_hat_gap"], bins=bins)
        ax.fill_between(
            edges.repeat(2)[1:-1],
            -np.repeat(counts_G, 2),
            step="pre",
            alpha=0.7,
            color="#6FB7E9",
            edgecolor="black"
        )

        ax.axvline(3, color="red", linestyle="--", linewidth=2, label="True K = 3")

        ax.set_title(rf"$\delta = {delta}$", fontsize=14)
        ax.set_xticks(np.arange(Kmin, Kmax + 1))
        ax.set_xlabel(r"$\widehat{K}$", fontsize=12)

        ymax = max(counts_F.max(), counts_G.max())
        ax.set_ylim(-1.1 * ymax, 1.1 * ymax)

        if idx in [0, 3]:   # first column of each row
            ax.set_ylabel("Frequency", fontsize=12)

    # ---- Legend in the top-right panel ----
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor="#3A8E7A", edgecolor="black", label="RC(3)"),
        Patch(facecolor="#6FB7E9", edgecolor="black", label="Gap Statistic")
    ]
    trueK_handle = Line2D(
        [0], [0],
        color="red",
        linestyle="--",
        linewidth=2,
        label="$K^* = 3$"
    )

    legend_ax_top.legend(
        handles=legend_handles + [trueK_handle],
        loc="center",
        fontsize=14
    )

    # y-axis positive ticks
    for ax in plot_axes:
        yticks = ax.get_yticks()
        ax.set_yticklabels([f"{int(abs(y))}" for y in yticks])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig5_n30_2x3.png"), bbox_inches="tight")
    plt.close()
