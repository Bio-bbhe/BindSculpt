import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set_theme(style="whitegrid", palette="viridis")

# 定义根目录和输出路径
ROOT_DIR = "/media/Data/hebeibei/Share/outputs/3bik_PD-1_truncated"
# RUN_FOLDERS = ["run_001_temp2.0", "run_002_temp1.0", "run_003_temp1.0"]
# 自动查找 run 文件夹（以 run_ 开头）
RUN_FOLDERS = sorted([
    folder for folder in os.listdir(ROOT_DIR)
    if os.path.isdir(os.path.join(ROOT_DIR, folder)) and folder.startswith("run_")
])

print(f"自动识别到 {len(RUN_FOLDERS)} 个 Run 文件夹：")
for run in RUN_FOLDERS:
    print(f" - {run}")


OUTPUT_CSV = os.path.join(ROOT_DIR, "pd1_binder_all_combined.csv")
PLOT_DIR = os.path.join(ROOT_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# 用于汇总所有数据
all_data_frames = []

print("开始读取并汇总数据...")

for folder in RUN_FOLDERS:
    run_id = folder
    full_path = os.path.join(ROOT_DIR, folder)
    files_to_load = {
        "iteration": os.path.join(full_path, "al_iteration_state.csv"),
        "filtered": os.path.join(full_path, "al_filtered_state.csv"),
        "raw_af3_success": os.path.join(full_path, "al_state.csv")
    }

    for source_type, file_path in files_to_load.items():
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if df.empty:
                    print(f"{file_path} 文件为空，跳过。")
                    continue

                if source_type == "raw_af3_success":
                    original_count = len(df)
                    df = df[df["af3_status"] == "success"].copy()
                    if df.empty:
                        print(f"{file_path} 中没有 af3_status 为 'success' 的条目，跳过。")
                        continue
                    print(f"从 {file_path} 读取 {original_count} 条，保留 {len(df)} 条 (af3_status=success)。")
                else:
                    print(f"从 {file_path} 读取 {len(df)} 条。")

                df["source"] = source_type
                df["run"] = run_id
                all_data_frames.append(df)
            except Exception as e:
                print(f"读取 {file_path} 时发生错误: {e}")
        else:
            print(f"文件不存在: {file_path}")

# 合并数据
if not all_data_frames:
    print("没有有效的数据读取成功，终止流程。")
    exit()

df_combined = pd.concat(all_data_frames, ignore_index=True)
print(f"\n数据汇总完成！共计 {len(df_combined)} 条记录。")

# 保存合并数据
df_combined.to_csv(OUTPUT_CSV, index=False)
print(f"合并数据已保存至：{OUTPUT_CSV}")

# 转换数值型字段
numeric_columns = [
    "ranking_score", "iptm", "ptm", "complex_plddt", "binder_plddt",
    "target_plddt", "shape_complementarity", "surface_hydrophobicity",
    "interface_delta_unsat_hbonds", "pae", "mpnn_score"
]
for col in numeric_columns:
    if col in df_combined.columns:
        df_combined[col] = pd.to_numeric(df_combined[col], errors="coerce")

# 提取 filtered 数据
df_filtered_only = df_combined[df_combined["source"] == "filtered"].copy()

# --- Top 5 最佳表现者 ---
print("\n Top 5 最佳表现者（按每个 run）")
for run in RUN_FOLDERS:
    df_run_filtered = df_filtered_only[df_filtered_only["run"] == run]
    if df_run_filtered.empty:
        print(f"\n{run} 的 filtered 数据为空，跳过。")
        continue

    print(f"\n🔹 Run: {run}")

    for metric in ["ranking_score", "iptm", "complex_plddt"]:
        if metric in df_run_filtered.columns:
            print(f"\n按 {metric} (越高越好):")
            cols_to_show = [col for col in ['id', 'ranking_score', 'iptm', 'complex_plddt', 'binder_sequence', 'mutation_type', 'mpnn_score'] if col in df_run_filtered.columns]
            print(df_run_filtered.sort_values(by=metric, ascending=False).head(5)[cols_to_show])

# --- 分布图绘制函数 ---
def plot_violin(data, x, y, hue, title, filename):
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=data, x=x, y=y, hue=hue, inner="quartile")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y.replace('_', ' ').title())
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    return path

def plot_hist(data, x, hue, title, filename, bins=30):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x=x, hue=hue, bins=bins, kde=True, multiple="stack")
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel("频数")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    return path

def plot_mutation_bar(data, filename):
    if "mutation_type" not in data.columns:
        print("无 mutation_type 字段，跳过柱状图。")
        return
    subset = data[data["source"].isin(["iteration", "filtered"])]
    summary = subset.groupby(["source", "mutation_type"]).size().reset_index(name="count")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=summary, x="mutation_type", y="count", hue="source")
    plt.title("Mutation Type Distribution (Filtered & Iteration)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved: {path}")
    return path

# --- 分布图分析 ---

# Violin plots for key metrics
for metric in numeric_columns:
    if metric in df_combined.columns and df_combined[metric].notna().any():
        plot_violin(
            data=df_combined,
            x="source",
            y=metric,
            hue="run",
            title=f"{metric.replace('_', ' ').title()} by Stage & Run",
            filename=f"{metric}_violin.png"
        )

# mpnn_score histogram
if "mpnn_score" in df_combined.columns and df_combined["mpnn_score"].notna().any():
    plot_hist(
        data=df_combined,
        x="mpnn_score",
        hue="run",
        title="MPNN Score Distribution by Run",
        filename="mpnn_score_histogram.png"
    )

# mutation_type bar plot
plot_mutation_bar(df_combined, "mutation_type_bar.png")

print("\n所有分析和图表已生成完毕，保存在：", PLOT_DIR)
