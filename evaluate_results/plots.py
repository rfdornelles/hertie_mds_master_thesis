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
plt.savefig("overall_accuracy_barplot.png", dpi=300, transparent=True)
plt.show()

###############################################################################
# 2. Heat‑map table – task accuracies
###############################################################################
# Use only the four task columns + overall
task_cols = ["numeric", "boolean", "categorical", "open_textual", "overall_accuracy"]
heat_data = (
    df_tasks_metrics_accuracy
    .set_index("experiment")[task_cols]
    .sort_values("overall_accuracy", ascending=False)
)

fig, ax = plt.subplots(figsize=(12, 0.5 + 0.4 * len(heat_data)))
sns.heatmap(
    heat_data,
    annot=True,
    fmt=".2f",
    cbar=False,
    linewidths=0.5,
    ax=ax,
    cmap="YlGnBu",
    vmin=0, vmax=1,
)
ax.set_title("Accuracy by task and experiment", weight="bold", pad=12)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig("task_accuracy_heatmap.png", dpi=300, transparent=True)
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

labels = ["numeric", "boolean", "categorical", "open_textual"]
theta = radar_factory(len(labels))

N = len(df_tasks_metrics_accuracy)
cols = 3
rows = int(np.ceil(N / cols))
fig = plt.figure(figsize=(cols * 4, rows * 4))

for idx, row in df_tasks_metrics_accuracy.iterrows():
    values = row[labels].tolist()
    ax = fig.add_subplot(rows, cols, idx + 1, projection="radar")
    ax.plot(theta, values, linewidth=2)
    ax.fill(theta, values, alpha=0.25)
    ax.set_ylim(0, 1)
    title = "\n".join(textwrap.wrap(row["experiment"], width=20))
    ax.set_title(title, size=9, weight="bold", pad=15)

fig.suptitle("Per‑task accuracy radar – each experiment", weight="bold", y=1.02)
plt.tight_layout()
plt.savefig("radar_charts_per_experiment.png", dpi=300, transparent=True)
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
    plt.savefig(filename, dpi=300, transparent=True)
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
overall_tbl.to_html("overall_accuracy_table.html", index=False)
task_tbl.to_html("task_accuracy_table.html", index=False)
