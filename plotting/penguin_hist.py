import pandas as pd
import glob, os
import matplotlib.pyplot as plt
from collections import Counter
if __name__ =='__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_dir = os.path.join(base_dir, "results/raw/penguins")
    output_dir = os.path.join(base_dir, "results/figures")

    files = sorted(glob.glob(os.path.join(input_dir, "K_trial_*.csv")))

    if not files:
        raise FileNotFoundError(f"No trial CSVs found in {input_dir}")


    df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    #output_path = os.path.join(input_dir, "K_combined.csv")
    #df_all.to_csv(output_path, index=False)
    counter = Counter(df_all['K_hat'])
    print(counter)
    plt.figure(figsize=(8, 5))
    plt.hist(df_all["K_hat"], bins=range(1, max(df_all["K_hat"]) + 2),
             color="#4A90E2", edgecolor="black", alpha=0.7)
    plt.axvline(x=3, color="red", linestyle="--", linewidth=2, label="True K = 3")
    plt.xlabel("Estimated K")
    plt.ylabel("Frequency")
    plt.title("Distribution of Estimated K across Trials")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #plt.savefig(os.path.join(output_dir, "penguin_hist2.png"))
    plt.show()