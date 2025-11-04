import sys, os
sys.path.append(os.path.abspath('../src'))
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    output_dir = os.path.join("../results/figures")
    type1_gao = pd.read_csv("../results/raw/type1_gao_by_repeat-2.csv")
    type1_gao_c = pd.read_csv("../results/raw/type1_gao_c_by_repeat.csv")
    type1_barber = pd.read_csv("../results/raw/type1_barber_by_repeat-2.csv")
    type1 = pd.read_csv("../results/raw/type1_error_randomized.csv")

    alpha = 0.05

    df_tau = type1.copy()
    df_tau.loc[df_tau['Type'].str.lower() == 'naive', 'Tau'] = 0.0
    df_tau['Group'] = df_tau['Tau'].astype(float).map(lambda x: f"{x:g}")
    df_tau_plot = df_tau[['Group', 'Type', 'Type I Error']]

    df_gb_plot = pd.concat([
        type1_gao.assign(Group='Gao_all', Type='Gao_all')[['Group', 'Type', 'Type I Error']],
        type1_gao_c.assign(Group='Gao_clustered', Type='Gao_clustered')[['Group', 'Type', 'Type I Error']],
        type1_barber.assign(Group='Barber', Type='Barber')[['Group', 'Type', 'Type I Error']]
    ], ignore_index=True)

    df_all = pd.concat([df_tau_plot, df_gb_plot], ignore_index=True)


    def _is_float(s):
        try:
            float(s)
            return True
        except:
            return False


    tau_groups = sorted([g for g in df_all['Group'].unique() if _is_float(g)], key=lambda s: float(s))
    extra_groups = [g for g in ['Barber', 'Gao_all', 'Gao_clustered'] if g in df_all['Group'].unique()]
    order = tau_groups + extra_groups

    green_shades = ["#C4EAA7", "#A9D595", "#8DBE7E", "#729869", "#587450", "#3F5237", "#252D1D"]
    palette = {
        '0': "#FF758F",
        'Gao_all': "#F7B718",
        'Gao_clustered':  "#8e1b01",
        'Barber': "#B069DB"
    }

    tau_values = sorted([float(t) for t in tau_groups])
    for t, c in zip([v for v in tau_values if v != 0], green_shades):
        palette[f"{t:g}"] = c

    df_all['Group'] = pd.Categorical(df_all['Group'], categories=order, ordered=True)

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=df_all, x='Group', y='Type I Error',
        hue='Group', order=order, palette=palette,
        showfliers=False, legend=False
    )

    ticks = ax.get_xticks()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    new_labels = []
    for lbl in labels:
        try:
            val = float(lbl)
            if val == 0:
                new_labels.append("Naive")
            else:
                new_labels.append(f"RAC({val:g})")
        except ValueError:
            new_labels.append(lbl)

    ax.set_xticks(ticks)
    ax.set_xticklabels(new_labels)

    plt.axhline(alpha, linestyle='--', linewidth=1, color='red')
    plt.xlabel('Method')
    plt.title('Type I Error by Method')
    #plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "typeI_combined.png"), bbox_inches='tight')
    plt.close()