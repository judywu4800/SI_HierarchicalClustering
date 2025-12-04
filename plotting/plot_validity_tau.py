import numpy as np
import sys, os
sys.path.append(os.path.abspath('../../src'))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_pvalues_from_folder(folder):
    files = sorted([f for f in os.listdir(folder) if f.startswith("trial_")])
    p_sel, p_unadj, p_naive = [], [], []

    for f in files:
        raw = open(os.path.join(folder, f)).read().strip()

        cleaned = (
            raw.replace("np.float64", "")
               .replace("(", "")
               .replace(")", "")
               .replace("[", "")
               .replace("]", "")
        )

        parts = cleaned.replace(" ", ",").split(",")

        parts = [p for p in parts if p.strip() != ""]

        vals = [float(p) for p in parts]

        if len(vals) != 3:
            raise ValueError(f"Unexpected p-value format in {f}: {raw}")

        p_sel.append(vals[0])
        p_unadj.append(vals[1])
        p_naive.append(vals[2])

    return np.array(p_sel), np.array(p_unadj), np.array(p_naive)


def plot_ecdf(p_sel, p_unadj, p_naive):
    plt.figure(figsize=(7,5))
    sns.ecdfplot(p_sel, label="Corrected for tau", color="blue")
    sns.ecdfplot(p_unadj, label="Unadjusted", color="green")
    sns.ecdfplot(p_naive, label="Naive", color="orange")
    plt.plot([0,1],[0,1],'--',color='red',label="Uniform(0,1)")
    plt.xlabel("P-value")
    plt.ylabel("ECDF")
    plt.title("Empirical CDF of P-values under Null")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.show()

def plot_qq(p_sel, p_unadj, p_naive):
    plt.figure(figsize=(7,5))
    n = len(p_sel)
    x = (np.arange(1, n+1) - 0.5) / n

    p_sel_q = np.sort(p_sel)
    p_unadj_q = np.sort(p_unadj)
    p_naive_q = np.sort(p_naive)

    plt.plot(x, p_sel_q, 'o', label="Corrected for tau", alpha=0.6)
    plt.plot(x, p_unadj_q, 'o', label="Unadjusted", alpha=0.6)
    plt.plot(x, p_naive_q, 'o', label="Naive", alpha=0.6)

    plt.plot([0,1],[0,1],'--', color='red', label="Uniform(0,1)")
    plt.xlabel("Theoretical Quantiles (Uniform)")
    plt.ylabel("Empirical Quantiles")
    plt.title("P-value QQ Plot")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.show()


if __name__ == "__main__":
    folder = "../results/raw/validity_tau_epsilon001"

    p_sel, p_unadj, p_naive = load_pvalues_from_folder(folder)

    plot_ecdf(p_sel, p_unadj, p_naive)
    plot_qq(p_sel, p_unadj, p_naive)