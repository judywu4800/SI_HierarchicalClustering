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

    linkages = ["Complete",  "Average", "Single","Minimax"]
    Ks = [2, 3]

    colors = {
        "sel": "#7B9669",     # RAC
        "gao": "#F7B718",     # Gao
        "barber": "#B069DB"   # Barber
    }

    base_path = "../results/raw/fig6_es"
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)']

    handles_for_legend = []
    labels_for_legend = []

    for row_idx, K in enumerate(Ks):
        for col_idx, linkage in enumerate(linkages):
            i = row_idx * len(linkages) + col_idx
            ax = axes[row_idx, col_idx]

            ax.set_title(f"{letters[i]} {linkage} (K={K})", fontsize=12)
            ax.set_xlabel("Effect size", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel("Power")
            ax.grid(True, linestyle="--", alpha=0.4)

            # File paths
            f_gao = f"{base_path}/rejection_es_gao_K{K}_{linkage}.csv"
            f_barber = f"{base_path}/rejection_es_barber_K{K}_{linkage}.csv"
            f_sel = f"{base_path}/reject_effect_size_K{K}_{linkage}.csv"

            # --- Gao curve ---
            if os.path.exists(f_gao):
                dfg = pd.read_csv(f_gao)
                bx, by, lower, upper, bc = binned_empirical_power_with_ci_normal(
                    dfg, xcol="effect_size", ycol="reject", n_bins=10, min_count=1, alpha=0.05
                )
                if len(bx) > 0:
                    latex_label_gao = r"Gao et al. ($\widehat{\sigma}$)"
                    h = ax.errorbar(
                        bx, by, yerr=[by - lower, upper - by],
                        fmt='s--', capsize=3, color=colors["gao"],
                        label=latex_label_gao
                    )[0]
                    if latex_label_gao not in labels_for_legend:
                        handles_for_legend.append(h)
                        labels_for_legend.append(latex_label_gao)
            else:
                print(f"Missing Gao file for K={K}, {linkage}")

            # --- Barber curve ---
            if os.path.exists(f_barber):
                dfb = pd.read_csv(f_barber)
                bx, by, lower, upper, bc = binned_empirical_power_with_ci_normal(
                    dfb, xcol="effect_size", ycol="reject", n_bins=10, min_count=1, alpha=0.05
                )
                if len(bx) > 0:
                    h = ax.errorbar(
                        bx, by, yerr=[by - lower, upper - by],
                        fmt='d--', capsize=3, color=colors["barber"],
                        label="Yun & Barber"
                    )[0]
                    if "Yun & Barber" not in labels_for_legend:
                        handles_for_legend.append(h)
                        labels_for_legend.append("Yun & Barber")
            else:
                print(f"Missing Barber file for K={K}, {linkage}")

            # --- Randomized (τ=0.1) ---
            if os.path.exists(f_sel):
                dfr = pd.read_csv(f_sel)
                dfr = dfr[dfr["tau"] == 0.1]
                if not dfr.empty:
                    bx, by, lower, upper, bc = binned_empirical_power_with_ci_normal(
                        dfr, xcol="effect_size", ycol="reject", n_bins=10, min_count=1, alpha=0.05
                    )
                    if len(bx) > 0:
                        h = ax.errorbar(
                            bx, by, yerr=[by - lower, upper - by],
                            fmt='o-', capsize=3, color=colors["sel"],
                            label="RAC (0.1)"
                        )[0]
                        if "RAC (0.1)" not in labels_for_legend:
                            handles_for_legend.append(h)
                            labels_for_legend.append("RAC (0.1)")
                else:
                    print(f"No tau=0.1 data for K={K}, {linkage}")
            else:
                print(f"Missing RAC file for K={K}, {linkage}")


            # --- Fallback if no data ---
            if len(ax.lines) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        fontsize=11, color="gray", transform=ax.transAxes)

            ax.set_ylim(-0.01, 1.05)
    all_handles, all_labels = [], []
    for ax in axes.flatten():
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)

    unique = dict(zip(all_labels, all_handles))

    desired_order = ["RAC (0.1)", r"Gao et al. ($\widehat{\sigma}$)", "Yun & Barber"]

    legend_labels = []
    legend_handles = []
    for label in desired_order:
        if label in unique:
            legend_labels.append(label)
            legend_handles.append(unique[label])

    print("Legend entries found:", legend_labels)

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.subplots_adjust(bottom=0.05)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        fontsize=10,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.01, 1, 0.97])
    save_path = os.path.join(output_dir, "fig6_power_K2_K3_linkages.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=400, pad_inches=0.02)
    plt.close()
    print(f"Saved figure to {save_path}")