import sys, os
sys.path.append(os.path.abspath('../src'))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm


def binned_empirical_power_with_ci_normal(df, xcol="effect_size", ycol="reject",
                                          n_bins=20, min_count=5, alpha=0.05):
    """Compute binned empirical power and normal-approx confidence intervals."""
    if df.empty:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    x_min, x_max = df[xcol].min(), df[xcol].max()
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


if __name__ == "__main__":
    output_dir = os.path.join("../results/figures")
    os.makedirs(output_dir, exist_ok=True)

    K = 2
    linkages = ["complete","single", "average",  "minimax"]

    colors = {
        "sel": "#587450",     # RAC
        "gao": "#F7B718",     # Gao
        "barber": "#B069DB"   # Barber
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    base_path = "../results/raw/fig6"

    letters = ['(a)', '(b)', '(c)', '(d)']

    for i, (ax, linkage) in enumerate(zip(axes, linkages)):
        ax.set_title(f"{letters[i]} {linkage} linkage", fontsize=12)
        ax.set_xlabel("Effect size")
        ax.set_ylabel("Power")
        ax.grid(True, linestyle="--", alpha=0.4)

        # File paths
        f_gao = f"{base_path}/rejection_es_gao_K{K}_{linkage}.csv"
        f_barber = f"{base_path}/rejection_es_barber_K{K}_{linkage}.csv"
        f_sel = f"{base_path}/reject_effect_size_K{K}_{linkage}.csv"

        has_gao = os.path.exists(f_gao)
        has_barber = os.path.exists(f_barber)
        has_sel = os.path.exists(f_sel)

        # --- Gao curve ---
        if has_gao:
            dfg = pd.read_csv(f_gao)
            bx, by, lower, upper, bc = binned_empirical_power_with_ci_normal(
                dfg, xcol="effect_size", ycol="reject",
                n_bins=10, min_count=1, alpha=0.05
            )
            if len(bx) > 0:
                ax.errorbar(
                    bx, by, yerr=[by - lower, upper - by],
                    fmt='s--', capsize=3, color=colors["gao"],
                    label="Gao et al. (σ_all)"
                )
        else:
            print(f"Missing Gao file for {linkage}")

        # --- Barber curve ---
        if has_barber:
            dfb = pd.read_csv(f_barber)
            bx, by, lower, upper, bc = binned_empirical_power_with_ci_normal(
                dfb, xcol="effect_size", ycol="reject",
                n_bins=10, min_count=1, alpha=0.05
            )
            if len(bx) > 0:
                ax.errorbar(
                    bx, by, yerr=[by - lower, upper - by],
                    fmt='d--', capsize=3, color=colors["barber"],
                    label="Yun & Barber"
                )
        else:
            print(f"Missing Barber file for {linkage}")

        # --- Randomized (τ=0.1) ---
        if has_sel:
            dfr = pd.read_csv(f_sel)
            dfr = dfr[dfr["tau"] == 0.1]
            if not dfr.empty:
                bx, by, lower, upper, bc = binned_empirical_power_with_ci_normal(
                    dfr, xcol="effect_size", ycol="reject",
                    n_bins=10, min_count=1, alpha=0.05
                )
                if len(bx) > 0:
                    ax.errorbar(
                        bx, by, yerr=[by - lower, upper - by],
                        fmt='o-', capsize=3, color=colors["sel"],
                        label="RAC (τ=0.1)"
                    )
            else:
                print(f"No tau=0.1 data for {linkage}")
        else:
            print(f"Missing RAC file for {linkage}")

        # --- Fallback: if nothing plotted ---
        if len(ax.lines) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    fontsize=12, color="gray", transform=ax.transAxes)

        # --- Axis range ---
        x_ranges = []
        for f in [f_gao, f_barber, f_sel]:
            if os.path.exists(f):
                try:
                    df_tmp = pd.read_csv(f)
                    if "effect_size" in df_tmp.columns:
                        x_ranges.append(df_tmp["effect_size"])
                except Exception:
                    pass

        x_min = max(dfg["effect_size"].min(), dfr["effect_size"].min(),
                    dfb["effect_size"].min())-0.1
        x_max = min(dfg["effect_size"].max(), dfr["effect_size"].max(),
                    dfb["effect_size"].max())+0.1
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(-0.01, 1)
        ax.legend(fontsize=10)

    #plt.suptitle(f"Power vs Effect Size (K={K})", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"fig6_linkages_K{K}.png"))
    plt.close()
