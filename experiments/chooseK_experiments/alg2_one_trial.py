import sys, os
sys.path.append(os.path.abspath('../../src'))
from alg2_one_trial2 import *


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--p", type=int, required=True)
    parser.add_argument("--delta", type=int, required=True)
    parser.add_argument("--K", type=int, required=True)
    parser.add_argument("--trial", type=int, required=True)
    args = parser.parse_args()
    np.random.seed(0)
    random.seed(0)
    n = args.n
    delta=args.delta
    p=args.p
    K = args.K
    #X, y = make_blobs(n_samples=n, n_features=2, centers=K, cluster_std=0.5, center_box=(-20,20), random_state=args.trial + 1000*K)
    X, labels, _ = generate_Kcluster_equal(n=n, p=p, K=K, delta=delta, sigma=1.0, seed = K + 1000*args.trial+6)
    alpha_list = generate_alpha_list_exp(n, 0.05, decay_rate=0.5)
    K_hat_F, p_values, alpha_seq, _ = find_best_K_F(X, tau=0.1, alpha_list=alpha_list,
                                     total_alpha=0.05, n_threshold=0.2*n, hard_threshold=0.05*n, seed = K + 1000*args.trial+6)
    K_hat_gap = gap_statistic(X)

    out = pd.DataFrame([[args.K, args.trial, K_hat_F, K_hat_gap]],
                       columns=["K_true", "trial", "K_hat_F", "K_hat_gap"])

    os.makedirs(f"results/k_hat_raw_K_n{n}_p{p}_delta{delta}", exist_ok=True)
    out.to_csv(f"results/k_hat_raw_K_n{n}_p{p}_delta{delta}/K{args.K}_trial{args.trial}.csv",
               index=False)

    df = pd.DataFrame({
        "pval": p_values,
        "alpha": alpha_seq
    })
    df["reject"] = df["pval"] < df["alpha"]

    os.makedirs(f"results/k_hat_raw_K_n{n}_p{p}_delta{delta}/pvals", exist_ok=True)
    df.to_csv(f"results/k_hat_raw_K_n{n}_p{p}_delta{delta}/pvals/K{args.K}_trial{args.trial}.csv",)



