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
    type1 = pd.read_csv("../results/raw/type1_error_results-3.csv")

    alpha = 0.05

    df_tau = type1.copy()
    df_tau.loc[df_tau['Type'].str.lower() == 'naive', 'Tau'] = 0.0
    df_tau['Group'] = df_tau['Tau'].astype(float).map(lambda x: f"{x:g}")
    df_tau_plot = df_tau[['Group', 'Type', 'Type I Error']]

    # --- FIX: make Type='Gao' for both, and distinct Group labels
    df_gb_plot = pd.concat([
        type1_gao.assign(Group='Gao_all', Type='Gao_all')[['Group', 'Type', 'Type I Error']],
        type1_gao_c.assign(Group='Gao_clustered', Type='Gao_clustered')[['Group', 'Type', 'Type I Error']],
        type1_barber.assign(Group='Barber', Type='Barber')[['Group', 'Type', 'Type I Error']]
    ], ignore_index=True)

    df_all = pd.concat([df_tau_plot, df_gb_plot], ignore_index=True)


    def _is_float(s):
        try:
            float(s);
            return True
        except:
            return False


    tau_groups = sorted([g for g in df_all['Group'].unique() if _is_float(g)],
                        key=lambda s: float(s))

    extra_groups = [g for g in ['Barber', 'Gao_all', 'Gao_clustered']
                    if g in df_all['Group'].unique()]
    order = tau_groups + extra_groups

    hue_order = [h for h in ['Randomized', 'Naive', 'Gao_all', 'Gao_clustered', 'Barber'] if
                 h in df_all['Type'].unique()]
    palette = {
        'Randomized': '#FF7F00',
        'Naive': '#377EB8',
        'Gao_all': '#4DAF4A',
        'Gao_clustered': '#E41A1C',
        'Barber': '#984EA3',
    }

    df_all['Group'] = pd.Categorical(df_all['Group'], categories=order, ordered=True)
    df_all['Type'] = pd.Categorical(df_all['Type'], categories=hue_order, ordered=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_all, x='Group', y='Type I Error',
        hue='Type', order=order, hue_order=hue_order,
        palette=palette, showfliers=False
    )
    plt.axhline(alpha, linestyle='--', linewidth=1, color='red', label=f'alpha={alpha}')
    plt.xlabel('tau / method')
    plt.title('Type I Error by tau and method')
    plt.legend(title='Type')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "typeI_combined.png"), bbox_inches='tight')
