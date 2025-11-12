import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":

    # ==== Step 1: 设置路径并读取数据 ====
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    n, p, equisized, iso = 30, 2, False, False
    input_dir = os.path.join(base_dir, "results/k_hat", f"k_hat_boxplot_n={n}_p={p}_equi{equisized}_iso{iso}")
    input_file = os.path.join(input_dir, "k_hat_results.csv")

    df = pd.read_csv(input_file)
    print(df.head())

    # ==== Step 2: 转换为长格式 (方便 hue 分组) ====
    df_long = df.melt(
        id_vars="delta",
        value_vars=["K_hat_F", "K_hat_gap"],
        var_name="Method",
        value_name="K_hat"
    )

    # 映射更友好的方法名
    df_long["Method"] = df_long["Method"].map({
        "K_hat_F": "Proposed Method",
        "K_hat_gap": "Gap Test"
    })

    # ==== Step 3: 绘制 side-by-side 箱线图 ====
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="delta", y="K_hat", hue="Method",
        data=df_long, palette=["skyblue", "lightcoral"], width=0.6
    )

    # 添加辅助线和标签
    plt.axhline(y=3, color="red", linestyle="--", linewidth=2)
    plt.xlabel("Cluster separation δ", fontsize=12)
    plt.ylabel("Estimated K̂", fontsize=12)
    plt.title("Comparison of Estimated Clusters: Proposed Method vs Gap Test", fontsize=13)
    plt.legend(title="Method", loc="upper left")
    plt.tight_layout()

    # ==== Step 4: 保存图像 ====
    output_path = os.path.join(input_dir, "k_hat_boxplot_sidebyside.png")
    #plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"✅ Figure saved to: {output_path}")
