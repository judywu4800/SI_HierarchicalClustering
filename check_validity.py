from utils import *
from datetime import datetime

if __name__ == "__main__":
    import os

    n = 30
    p = 10
    sigma = 1.0
    K = 3
    tau_list = [0.05, 0.1,0.5, 1.0, 3.0,5.0,10]
    layer = -1
    linkage = "complete"
    num_trials = 500
    n_jobs = -1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"pval_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    all_p_values, naive_p_values = check_p_value_uniformity_multi_tau_parallel(
        n, p, sigma, K, tau_list, layer, linkage, num_trials, n_jobs
    )

    # Save raw data
    df = pd.DataFrame({f"tau={tau}": all_p_values[tau] for tau in tau_list})
    df["naive"] = naive_p_values
    df.to_csv(os.path.join(output_dir, "pval_data.csv"), index=False)

    '''
    plt.figure(figsize=(10, 6))
    for tau in tau_list:
        plt.hist(all_p_values[tau], bins=20, density=True, alpha=0.4,
                 label=f"Sel. (tau={tau})", edgecolor='black')
    plt.hist(naive_p_values, bins=20, density=True, alpha=0.4,
             label=f"Naive", edgecolor='gray', linestyle='dashed')
    plt.axhline(1, color='red', linestyle='dashed', linewidth=2, label="Uniform(0,1)")
    plt.xlabel("P-value")
    plt.ylabel("Density")
    plt.title("Histogram of P-values Under the Null")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "histogram_pvalues.png"))
    plt.close()
    '''


    plt.figure(figsize=(10, 6))
    for tau in tau_list:
        sns.ecdfplot(all_p_values[tau], label=f"Sel. (tau={tau})", linestyle="-")
    sns.ecdfplot(naive_p_values, label="Naive", linestyle="--")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Expected (Uniform)")
    plt.xlabel("P-value")
    plt.ylabel("ECDF")
    plt.title("ECDF of P-values")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ecdf_pvalues.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    theoretical_quantiles = np.linspace(0, 1, num_trials)
    for tau in tau_list:
        sorted_sel = np.sort(all_p_values[tau])
        plt.plot(theoretical_quantiles, sorted_sel, marker='o', linestyle='', label=f"Sel. (tau={tau})")
    sorted_naive = np.sort(naive_p_values)
    plt.plot(theoretical_quantiles, sorted_naive, marker='x', linestyle='', label="Naive")
    plt.plot([0, 1], [0, 1], linestyle="--", color="red", label="Expected (Uniform)")
    plt.xlabel("Theoretical Uniform Quantiles")
    plt.ylabel("Empirical P-values")
    plt.title("Q-Q Plot: P-values vs. Uniform(0,1)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "qq_plot_pvalues.png"))
    plt.close()
