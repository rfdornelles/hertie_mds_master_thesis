# ----------------------------------------------------------------------------
# Imports & style
# ----------------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams["font.family"] = "DejaVu Sans"

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
df_all = pd.read_csv("all_metrics.csv")
df_experiments = df_all["experiment"].copy().to_frame()

def get_model_name(exp):
    s = exp.lower()

    if "lawma" in s:
        return "Lawma 8B"
    if "tucano" in s:
        return "Tucano"
    if "llama" in s:
        if "3.2" in s:
            return "Llama 3.2"
        elif "3.1" in s:
            return "Llama 3.1"
    if "phi" in s:
        return "Phi 4"
    if "gemma" in s:
        return "Gemma 3"
    if "o3-mini" in s:
        return "o3-mini"
    if "gpt" in s and "4.1" in s:
        if "nano" in s:
            return "GPT 4.1-nano"
        elif "mini" in s:
            return "GPT 4.1-mini"
        else:
            return "GPT 4.1"
    if "4o" in s:
        if "mini" in s:
            return "GPT 4o-mini"
        else:
            return "GPT 4o"
    return "Unknown"

df_experiments["model"] = df_experiments["experiment"].apply(get_model_name)

# Classify family using np.select (like case_when)
conditions_fam = [
    df_experiments["experiment"].str.contains('open_ai', case=False),
    df_experiments["experiment"].str.contains('llama', case=False),
    df_experiments["experiment"].str.contains('gemma', case=False),
    df_experiments["experiment"].str.contains('phi', case=False),
    df_experiments["experiment"].str.contains('lawma', case=False),
    df_experiments["experiment"].str.contains('tucano', case=False)
]
choices_fam = ['OpenAI', 'Llama', 'Gemma', 'Phi', 'Lawma', 'Tucano']
df_experiments['family'] = np.select(conditions_fam, choices_fam, default='ERROR')

# Classify parameters using np.select (like case_when)
conditions_params = [
    df_experiments['family'] == 'OpenAI',
    df_experiments['family'] == 'Lawma',
    df_experiments['family'] == 'Tucano',
    df_experiments["experiment"].str.contains('12b', case=False),
    df_experiments["experiment"].str.contains('1b', case=False),
    df_experiments["experiment"].str.contains('27b', case=False),
    df_experiments["experiment"].str.contains('8b', case=False),
    df_experiments["experiment"].str.contains('3b', case=False),
    df_experiments["experiment"].str.contains('14b', case=False)
]
choices_params = [
    'unknown',  # For OpenAI
    '8 B',      # For Lawma
    '2.4 B',    # For Tucano
    '12 B',
    '1 B',
    '27 B',
    '8 B',
    '3 B',
    '14 B'
]
df_experiments['parameters'] = np.select(conditions_params, choices_params, default='ERROR')

# Classify if finetuned or base model
df_experiments['status'] = np.where(
    df_experiments['experiment'].str.contains('ft|finetune', case=False),
    'finetuned',
    'base model'
)

# Classify if commercial or open-source based on the presence of 'open_ai'
df_experiments['type'] = np.where(
    df_experiments['experiment'].str.contains('open_ai', case=False),
    'commercial',
    'open-source'
)

# merge (remove stale cols first)
for col in ["model", "family", "parameters", "type", "status"]:
    df_all.drop(columns=col, errors="ignore", inplace=True)

df_all = df_all.merge(
    df_experiments,
    on="experiment",
    how="left"
)

df_all.sort_values(["status", "type", "model", "accuracy"], ascending=[True, True, True, False], inplace=True)
df_all.reset_index(drop=True, inplace=True)

unique_models = df_all[['model', 'family', 'parameters', 'type']].drop_duplicates()
unique_models.to_csv("model_metadata.csv", index=False)

df_all.to_csv("accuracy_metadata.csv", index=False)
# ----------------------------------------------------------------------------
# 2. Utilities – table & heatmap PNG
# ----------------------------------------------------------------------------

def _apply_zebra(table, even="#ffffff", odd="#f7f7f7"):
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # header
            cell.set_facecolor("#eaeaea")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor(even if row % 2 == 0 else odd)


def save_table_png(df: pd.DataFrame, path: str, title: str, fontsize: int = 8):
    # Reduced figure size for a smaller table output
    fig, ax = plt.subplots(figsize=(df.shape[1] * 1.0, 0.4 * len(df) + 0.8))
    fig.patch.set_alpha(0)
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1, 1)  # Adjusted scale for a more compact table
    _apply_zebra(tbl)
    ax.set_title(title, weight="bold")
    plt.tight_layout()
    fig.savefig(path, dpi=300, transparent=True)
    plt.close(fig)


def save_heatmap(df: pd.DataFrame, path: str, title: str, fmt: str = ".3f", cmap: str = "YlGnBu"):
    fig, ax = plt.subplots(figsize=(3.0 + df.shape[1] * 1.2, 0.46 * len(df) + 1.6))
    fig.patch.set_alpha(0)
    sns.heatmap(df, annot=True, fmt=fmt, cmap=cmap, cbar=False,
                linewidths=0.4, linecolor="#e0e0e0", ax=ax)
    ax.set_title(title, pad=14, weight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(path, dpi=300, transparent=True)
    plt.close(fig)

# ----------------------------------------------------------------------------
# 3. Table 1 – Descriptive metadata
# ----------------------------------------------------------------------------

save_table_png(unique_models, "model_metadata_table.png", "Model catalogue – descriptive metadata", fontsize=10)

# ----------------------------------------------------------------------------
# 4. Table 2 – Overall accuracy heat‑map (all models, 3 d.p.)
# ----------------------------------------------------------------------------
acc_df = df_all[df_all["status"] == "base model"][["model", "accuracy"]]
acc_df["accuracy"] = acc_df["accuracy"].round(3)
acc_df = acc_df.sort_values("accuracy", ascending=False).reset_index(drop=True)
heat_df = acc_df.set_index("model")[["accuracy"]]
save_heatmap(heat_df, "accuracy_heatmap.png", "Overall accuracy per model (3 d.p.)", fmt=".3f")

# ----------------------------------------------------------------------------
# 5. Table 3 – Fine‑tuning impact (paired models only)
# ----------------------------------------------------------------------------
wide = df_all.pivot_table(index=["model", "type"], columns="status", values="accuracy", aggfunc="first")
paired = wide.dropna(subset=["base model", "finetuned"]).copy()
paired["Δ abs"] = (paired["finetuned"] - paired["base model"]).round(3)
paired["Δ %"] = ((paired["Δ abs"] / paired["base model"]) * 100).round(1)
impact_tbl = (
    paired.reset_index()
    .rename(columns={"base model": "base model_acc", "finetuned": "ft_acc"})
    [["model", "type", "base model_acc", "ft_acc", "Δ abs", "Δ %"]]
    .sort_values("Δ abs", ascending=False)
    .reset_index(drop=True)
)
save_table_png(impact_tbl, "finetune_impact_table.png", "Fine‑tuning impact (absolute & relative gains)")

print("✅ PNGs ready: model_metadata_table.png | accuracy_heatmap.png | finetune_impact_table.png")
