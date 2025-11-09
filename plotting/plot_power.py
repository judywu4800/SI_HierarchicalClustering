import sys, os
sys.path.append(os.path.abspath('../src'))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import norm


if __name__ == "__main__":
    output_dir = os.path.join("../results/figures")

    #dfg = pd.read_csv("../results/raw/rejection_and_effect_gao.csv")
    #dfgc = pd.read_csv("../results/raw/rejection_and_effect_gao_clustered.csv")
    #dfb = pd.read_csv("../results/raw/rejection_and_effect_barber.csv")
    #dfr = pd.read_csv("../results/raw/reject_es-6.csv")
    dfg = pd.read_csv("../results/raw/rejection_es_gao_K2.csv")
    dfgc = pd.read_csv("../results/raw/rejection_es_gao_clustered_K2.csv")
    dfb = pd.read_csv("../results/raw/rejection_es_barber_K2.csv")
    dfr = pd.read_csv("../results/raw/reject_effect_size_K2.csv")



    def binned_empirical_power_with_ci_normal(df, xcol="effect_size", ycol="reject",
                                              x_min=1.0, x_max=3.0,
                                              n_bins=20, min_count=5,
                                              alpha=0.05):

        edges = np.linspace(x_min, x_max, n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        idx = pd.cut(df[xcol].to_numpy(), bins=edges,
                     include_lowest=True, right=True, labels=False)

        valid = ~pd.isna(idx)
        idx = idx[valid].astype(int)
        y = df.loc[valid, ycol].to_numpy()

        counts = np.bincount(idx, minlength=n_bins)
        sums = np.bincount(idx, weights=y, minlength=n_bins)

        prop = np.full(n_bins, np.nan)
        lower = np.full(n_bins, np.nan)
        upper = np.full(n_bins, np.nan)

        z = norm.ppf(1 - alpha / 2)

        for i in range(n_bins):
            if counts[i] >= min_count:
                p_hat = sums[i] / counts[i]
                se = np.sqrt(p_hat * (1 - p_hat) / counts[i])
                ci_low = max(0, p_hat - z * se)
                ci_up = min(1, p_hat + z * se)
                prop[i], lower[i], upper[i] = p_hat, ci_low, ci_up

        mask = ~np.isnan(prop)
        return centers[mask], prop[mask], lower[mask], upper[mask], counts[mask]


    colors = {
        "sel": ["#587450","#8DBE7E","#729869",  "#6BBBAF", "#3CA1A4"],#["#C4EAA7", "#A9D595", "#8DBE7E", "#729869", "#587450", "#3F5237", "#252D1D"],
        "gao": "#F7B718",
        "barber": "#B069DB"
    }

    plt.figure(figsize=(10, 6))

    n_bins = 10
    alpha = 0.05
    x_min = max(dfg["effect_size"].min(), dfr["effect_size"].min(),
                dfb["effect_size"].min())
    x_max = min(dfg["effect_size"].max(), dfr["effect_size"].max(),
                dfb["effect_size"].max())

    for i, (tau, g) in enumerate(dfr.groupby("tau")):
        if tau in [0,0.05,0.025,0.075, 0.5, 0.75, 1.0]:
            continue
        bx, by, lower, upper, bc = binned_empirical_power_with_ci_normal(
            g, xcol="effect_size", ycol="reject",
            x_min=x_min, x_max=x_max, n_bins=n_bins, min_count=3,
            alpha=alpha
        )
        plt.errorbar(
            bx, by, yerr=[by - lower, upper - by],
            fmt='o-', capsize=4,
            color=colors["sel"][i % len(colors["sel"])],
            label=f"RAC({tau})"
        )

    bx, by, lower, upper, bc = binned_empirical_power_with_ci_normal(
        dfgc, xcol="effect_size", ycol="reject",
        x_min=x_min, x_max=x_max, n_bins=n_bins, min_count=3,
        alpha=alpha
    )
    if len(bx) > 0:
        plt.errorbar(
            bx, by, yerr=[by - lower, upper - by],
            fmt='s--', capsize=4,
            color='red',
            label="Gao et al. (sigma_clustered)"
        )
    bx, by, lower, upper, bc = binned_empirical_power_with_ci_normal(
        dfg, xcol="effect_size", ycol="reject",
        x_min=x_min, x_max=x_max, n_bins=n_bins, min_count=3,
        alpha=alpha
    )
    if len(bx) > 0:
        plt.errorbar(
            bx, by, yerr=[by - lower, upper - by],
            fmt='s--', capsize=4,
            color=colors["gao"],
            label="Gao et al. (sigma_all)"
        )
    bx, by, lower, upper, bc = binned_empirical_power_with_ci_normal(
        dfb, xcol="effect_size", ycol="reject",
        x_min=x_min, x_max=x_max, n_bins=n_bins, min_count=3,
        alpha=alpha
    )
    if len(bx) > 0:
        plt.errorbar(
            bx, by, yerr=[by - lower, upper - by],
            fmt='d--', capsize=4,
            color=colors["barber"],
            label="Yun & Barber"
        )

    plt.xlabel("Effect size")
    plt.ylabel("Power (Pr[Reject=1 |Effect Size])")
    plt.title("Power vs Effect Size")
    plt.legend(title="Method / Tau")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xlim(x_min, x_max)
    plt.ylim(-0.01, 1)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "power_vs_effectsize_K2.png"))
    plt.close()