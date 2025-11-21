import sys, os
sys.path.append(os.path.abspath('../../src'))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from utils import generate_null_data
from find_best_K import find_best_K_F, generate_alpha_list_exp

def run_single_trial(tau, n, p, sigma, total_alpha):
    X_null = generate_null_data(n, p, np.zeros(p), sigma)
    alpha_list = generate_alpha_list_exp(n=n, total_alpha=total_alpha, decay_rate=0.5)
    K_hat, _, _, _ = find_best_K_F(X_null, tau=tau, alpha_list=alpha_list.copy(), total_alpha=total_alpha,n_threshold=0.4*n, hard_threshold=0.1*n)
    return int(K_hat > 1), K_hat

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--num_trials", type=int, default=2000)
    parser.add_argument("--n", type=int, default=90)
    parser.add_argument("--p", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1)
    parser.add_argument("--total_alpha", type=float, default=0.05)
    parser.add_argument("--outdir", type=str, default="../../results/raw/fwer")
    args = parser.parse_args()

    trial_results = Parallel(n_jobs=-1)(
        delayed(run_single_trial)(args.tau, args.n, args.p, args.sigma, args.total_alpha)
        for _ in range(args.num_trials)
    )

    errors, K_hats = zip(*trial_results)
    fwer = sum(errors) / args.num_trials
    tau_label = "naive" if args.tau == 0 else args.tau

    os.makedirs(args.outdir, exist_ok=True)

    pd.DataFrame([{
        "tau": tau_label,
        "FWER": fwer,
        "num_trials": args.num_trials
    }]).to_csv(os.path.join(args.outdir, f"fwer_tau_{tau_label}.csv"), index=False)

    pd.DataFrame({"K_hat": K_hats}).to_csv(
        os.path.join(args.outdir, f"khat_tau_{tau_label}.csv"), index=False
    )
