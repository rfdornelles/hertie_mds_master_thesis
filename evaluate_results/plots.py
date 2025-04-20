# ------------------------------------------
# Visualising evaluation results (refactored)
# ------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import textwrap
import pandas as pd
from IPython.display import display, HTML

sns.set_theme(style="whitegrid", palette="pastel")

## Import
df_all_metrics = pd.read_csv("all_metrics.csv")
df_experiments_score = pd.read_csv("experiments_score.csv")
df_tasks_metrics_accuracy = pd.read_csv("tasks_metrics_accuracy.csv")

#### data preparation
# classify experiments acording with: family, parameters, OS/commercial, ft/baseline
df_experiments = df_all_metrics["experiment"].copy().to_frame()

# add model name: GPT 4.1-nano, GPT 4.1-mini, GPT 4.1, GPT 4o, o3-mini, OpenAi 4o-mini, Phi 4.0, Gemma 4.0, Llama 3.1, Llama 3.2, Tucano 2.4B, Lawma 8B
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

## add classification to df_all_metrics
df_all_metrics = df_all_metrics.merge(
    df_experiments,
    on="experiment",
    how="left"
)

df_tasks_metrics_accuracy = df_tasks_metrics_accuracy.merge(
    df_experiments,
    on="experiment",
    how="left"
)


## pairs of experiments base - finetuned
# 3_open_ai_test_4o-mini - 7_open_ai_test_ft_4o_mini
# phi4_14b_baseline_unsloth - phi4_14b_finetune_unsloth
# llama_3.2_3b_baseline - llama_3.2_3B_ft_unsloth_2025-04-13_17-52-07

## Plot to compare base vs finetuned

# only keep the above experiments
df_experiments_score_comparison = df_tasks_metrics_accuracy[
    df_tasks_metrics_accuracy["experiment"].isin([
        "3_open_ai_test_4o-mini",
        "7_open_ai_test_ft_4o_mini",
        "phi4_14b_baseline_unsloth",
        "phi4_14b_finetune_unsloth",
        "llama_3.2_3b_baseline",
        "llama_3.2_3B_ft_unsloth_2025-04-13_17-52-07"
    ])
].copy().sort_values(["model", "status"])

# plot base vs finetuned
fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    data=df_experiments_score_comparison,
    x="model", y="overall_accuracy",
    hue="status",
    ax=ax
)
ax.set_xlabel("Model")
ax.set_ylabel("Overall accuracy")
ax.set_title("Base vs Finetuned accuracy", weight="bold")
ax.set_ylim(0, 1.0)
## add a line using gpt 4o-mini as reference
gpt_4o_mini_acc = df_experiments_score_comparison[
    df_experiments_score_comparison["experiment"] == "3_open_ai_test_4o-mini"
]["overall_accuracy"].values[0]
ax.axhline(y=gpt_4o_mini_acc, color='r', linestyle='--')
ax.text(
    x=0.5, y=gpt_4o_mini_acc + 0.02,
    s="GPT 4o-mini", color='r', ha='center', va='bottom'
)
# add legend
ax.legend(loc="upper center")
# add grid
ax.grid(axis='y', linestyle='--', alpha=0.7)
# add text to each bar
for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.3f}",
        (p.get_x() + p.get_width() / 2., p.get_height()),
        ha='center', va='bottom',
        fontsize=10,
        color='black',
        xytext=(0, 5),
        textcoords='offset points'
    )
# rotate x labels
# plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("base_vs_finetuned_accuracy.png", dpi=900, transparent=True)
plt.show()

## compare per-task accuracies for each model
# Filter the tasks dataset for the selected experiments
df_tasks_comparison = df_tasks_metrics_accuracy[
    df_tasks_metrics_accuracy["experiment"].isin([
        "3_open_ai_test_4o-mini",
        "7_open_ai_test_ft_4o_mini",
        "phi4_14b_baseline_unsloth",
        "phi4_14b_finetune_unsloth",
        "llama_3.2_3b_baseline",
        "llama_3.2_3B_ft_unsloth_2025-04-13_17-52-07"
    ])
].copy().sort_values(["model", "status"])



# List of tasks to compare
tasks = ["numeric", "boolean", "categorical", "open_textual"]

# Create subplots for each task in a 2x2 grid
fig, axs = plt.subplots(2, 2, figsize=(12, 10), sharey=True)
axs = axs.ravel()  # flatten the grid

for i, task in enumerate(tasks):
    ax = axs[i]
    sns.barplot(
        data=df_tasks_comparison,
        x="model", y=task, hue="status",
        ax=ax
    )
    ax.set_title(f"{task.capitalize()} accuracy", weight="bold")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Model")
    ax.tick_params(axis='x', rotation=45)
    if i % 2 == 0:
        ax.set_ylabel("Accuracy")
    else:
        ax.set_ylabel("")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Annotate each bar with its value
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width()/2., p.get_height()),
            ha='center', va='bottom',
            fontsize=9,
            color='black',
            xytext=(0, 5),
            textcoords='offset points'
        )
    # Remove individual legends
    if ax.get_legend() is not None:
        ax.get_legend().remove()

# Create one shared legend at the top center
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels,loc="upper center", ncol=len(labels))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("per_task_accuracy_comparison.png", dpi=900, transparent=True)
plt.show()

# plt.savefig("base_vs_finetuned_accuracy.png", dpi=300, transparent=True)



###############################################################################
# 1. Bar plot – overall accuracy per experiment
###############################################################################
fig, ax = plt.subplots(figsize=(10, 5))
sorted_acc = df_all_metrics.sort_values("accuracy", ascending=False)

sns.barplot(
    data=sorted_acc,
    y="experiment", x="accuracy",
    ax=ax
)
ax.set_xlabel("Overall accuracy")
ax.set_ylabel("Experiment")
ax.set_title("Overall accuracy by experiment", weight="bold")
ax.set_xlim(0, 1.0)
plt.tight_layout()
plt.savefig("overall_accuracy_barplot.png", dpi=900, transparent=True)
plt.show()

###############################################################################
# 2. Heat‑map table – task accuracies
###############################################################################
# Use only the four task columns + overall
task_cols = ["overall_accuracy", "numeric", "boolean", "categorical", "open_textual"]
heat_data = (
    df_tasks_metrics_accuracy
    .set_index("model")[task_cols]
    .sort_values("overall_accuracy", ascending=False)
)

# remove accuracy 0
heat_data = heat_data[heat_data["overall_accuracy"] > 0]


fig, ax = plt.subplots(figsize=(12, 0.5 + 0.4 * len(heat_data)))
sns.heatmap(
    heat_data,
    annot=True,
    fmt=".2f",
    cbar=False,
    linewidths=0.5,
    ax=ax,
    cmap="YlGnBu",
    vmin=0.3, vmax=0.95,
)
# x axis in the top
ax.xaxis.set_ticks_position("top")
ax.set_title("Accuracy by model and experiment", weight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("task_accuracy_heatmap.png", dpi=900, transparent=True)
plt.show()

###############################################################################
# 3. Radar charts – one subplot per experiment
###############################################################################
def radar_factory(num_vars, frame="circle"):
    """Return a function that creates a radar plot with *num_vars* axes.
    Source: Matplotlib docs, slightly trimmed."""
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection

    # calculate evenly‑spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = "radar"
        # use 1 line for each variable (dotted)
        draw_frame = False

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_offset(np.pi / 2)
            self.set_theta_direction(-1)
            self.set_thetagrids(np.degrees(theta), labels)

        def fill(self, *args, **kwargs):
            return super().fill(*args, **kwargs)

        def plot(self, *args, **kwargs):
            return super().plot(*args, **kwargs)

    register_projection(RadarAxes)
    return theta
df_tasks_metrics_accuracy_clean = df_tasks_metrics_accuracy[df_tasks_metrics_accuracy["overall_accuracy"] > 0].copy()


labels = ["numeric", "boolean", "categorical", "open_textual"]
theta = radar_factory(len(labels))

N = len(df_tasks_metrics_accuracy)
cols = 3
rows = int(np.ceil(N / cols))
fig = plt.figure(figsize=(cols * 4, rows * 4))

for idx, row in df_tasks_metrics_accuracy_clean.iterrows():
    values = row[labels].tolist()
    ax = fig.add_subplot(rows, cols, idx + 1, projection="radar")
    ax.plot(theta, values, linewidth=2)
    ax.fill(theta, values, alpha=0.25)
    ax.set_ylim(0, 1)
    title = "\n".join(textwrap.wrap(f"{row['model']}-{row['status']}", width=20))
    ax.set_title(title, size=9, weight="bold", pad=15)

fig.suptitle("Per‑task accuracy radar – each experiment", weight="bold", y=1.02)
plt.tight_layout()
plt.savefig("radar_charts_per_experiment.png", dpi=900, transparent=True)
plt.show()

###############################################################################
# 4. Static tables (PNG + HTML) – overall & per‑task
###############################################################################
def save_table_png(df, filename, title=""):
    """
    Render *df* as a matplotlib table and save it as a transparent PNG.
    """
    fig, ax = plt.subplots(figsize=(df.shape[1] * 2.2, 0.55 * len(df) + 1.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)
    if title:
        ax.set_title(title, weight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=900, transparent=True)
    plt.close()

# --------------------------------------------------------------------------
# Overall‑accuracy table (sorted)
overall_tbl = (
    df_all_metrics.sort_values("accuracy", ascending=False)
    .reset_index(drop=True)
    .round(3)
)
save_table_png(overall_tbl, "overall_accuracy_table.png", "Overall accuracy")

# --------------------------------------------------------------------------
# Per‑task accuracy table  (same order as heat‑map)
task_tbl = (
    heat_data.reset_index()
    .rename(columns={"index": "experiment"})
    .round(3)
)
save_table_png(task_tbl, "task_accuracy_table.png", "Accuracy by task")

# --------------------------------------------------------------------------
# Also export tidy HTML copies (great for slides or Confluence/Wiki)
# overall_tbl.to_html("overall_accuracy_table.html", index=False)
# task_tbl.to_html("task_accuracy_table.html", index=False)
