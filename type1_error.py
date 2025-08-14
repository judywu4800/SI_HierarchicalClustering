from utils import *
from datetime import datetime

if __name__ == "__main__":
    import os

    n = 30
    p = 10
    sigma = 1
    tau_list = [0,0.1, 0.25, 0.5, 1,1.5,2,5,10]
    #tau_list = [0,0.1]
    K = 3
    layer = -1
    alpha = 0.05
    num_trials = 200
    num_repeats = 100
    n_jobs = -1

    df_results = check_type1_multi_tau_random_pair_parallel(n, p, sigma, tau_list, K,
                                                 alpha=alpha, num_trials=num_trials,
                                                 num_repeats=num_repeats, n_jobs=n_jobs)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"results_type1_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    df_results.to_csv(os.path.join(output_dir, "type1_error_results.csv"), index=False)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_results, x="Tau", y="Type I Error", hue="Type")
    plt.axhline(y=alpha, linestyle='--', color='red', label=f"Significance level α = {alpha}")
    plt.title(f"Distribution of Type I Error Rates over {num_repeats} Repetitions (Layer {layer})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "type1_error_boxplot.png"))
    plt.close()