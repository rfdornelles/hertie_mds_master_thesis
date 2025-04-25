## Retrieve data from Weights and Biases API

from dotenv import load_dotenv
import os
import wandb
import json
from json import JSONDecodeError
import pandas as pd
import numpy as np

# Load environment variables and login to W&B
load_dotenv()
wandb.login()

# Initialize W&B API
api = wandb.Api()
entity = api.default_entity  # Automatically uses the logged-in entity

# Get all projects under the current entity
projects = api.projects(entity)

# Table to store combined data
table = []

# Loop over all projects
for project in projects:
    print(f"\n🔍 Project: {project.name}")
    runs = api.runs(f"{entity}/{project.name}")

    # Loop over all runs in the project
    for run in runs:
        run_dir = os.path.join(entity, project.name, run.id)
        os.makedirs(run_dir, exist_ok=True)

        # Define metadata and summary file paths
        metadata_path = os.path.join(run_dir, "metadata.json")
        summary_path = os.path.join(run_dir, "summary.json")

        # Skip run if both metadata and summary already exist
        if os.path.exists(metadata_path) and os.path.exists(summary_path):
            print(f"✅ Skipping existing run: {run.id}")
            continue

        print(f"⬇️  Downloading run: {run.name} ({run.id})")

        # Download all files associated with the run (from W&B Files tab)
        for file in run.files():
            file_path = os.path.join(run_dir, file.name)
            if not os.path.exists(file_path):
                file.download(root=run_dir, replace=True)

        # Prepare metadata
        # Handle created_at as string if not datetime
        created_at_value = run.created_at
        try:
            created_at_str = created_at_value.isoformat()
        except AttributeError:
            created_at_str = str(created_at_value)

        metadata = {
            "entity": entity,
            "project": project.name,
            "run_id": run.id,
            "run_name": run.name,
            "created_at": created_at_str,
            "state": run.state,
            "tags": run.tags,
            "url": run.url,
        }

        # Add run configuration (hyperparameters)
        for k, v in run.config.items():
            if not k.startswith("_"):
                metadata[f"param_{k}"] = v

        # Save metadata as JSON
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Collect and save summary metrics
        summary = {}
        for k, v in run.summary.items():
            # Ensure JSON serializable: convert non-primitive to string
            if isinstance(v, (int, float, str, bool)):
                summary[f"metric_{k}"] = v
            else:
                summary[f"metric_{k}"] = str(v)

        # Write summary JSON
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

# After all runs are processed, build the final table
print("\n📦 Building combined table from saved JSON files...")

def safe_load_json(path):
    """
    Load JSON from `path`, but return an empty dict if the file is missing,
    empty, or contains invalid JSON.
    """
    try:
        if os.path.getsize(path) == 0:
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, JSONDecodeError):
        return {}

# Walk through entity directory to aggregate all run data
table = []
for root, dirs, files in os.walk(entity):
    if "metadata.json" in files:
        meta = safe_load_json(os.path.join(root, "metadata.json"))
        summary = safe_load_json(os.path.join(root, "summary.json"))
        combined = {**meta, **summary}
        table.append(combined)

print(f"Processed {len(table)} entries.")

# Save combined table as CSV
df = pd.DataFrame(table)
df.to_csv("wandb_runs_table.csv", index=False)

print("\n✅ Final table saved as: wandb_runs_table.csv")
print(df.head())

# filter only finished runs
filtered_df = df[df['state'] == 'finished']
# only relative to ft not containing inference
filtered_df = filtered_df[~filtered_df['run_name'].str.contains(r'inference')]
filtered_df = filtered_df[filtered_df['run_name'].str.contains(r'ft|finetune')]
# order by metric_eval/runtime
filtered_df = filtered_df.sort_values(by='metric_eval/runtime', ascending=False)

interest_experiment = [
    'ft_hf_lora_Phi-4_v1_2025-04-15_14-00-56',
    'ft_lora_sabia-7b_v1_2025-04-25_17-01-57',
    'ft_unsloth__Llama-3.2-3B-Instruct_v1_2025-04-13_17-52-07'
]

## retrieve the run_ids of the interest_experiment
run_ids = filtered_df[filtered_df['run_name'].isin(interest_experiment)]['run_id'].tolist()

# collect all per-step rows here
rows = []


ENTITY   = "rodornelles-hertie-school"
PROJECT  = "master_thesis-open_source_models"
SAMPLES  = 10_000


TRAIN_KEYS = ["train/loss", "loss_train", "metric_loss_train"]
EVAL_KEYS  = ["eval/loss",  "loss_valid", "metric_loss_valid"]

for run_id in run_ids:
    run = api.run(f"{ENTITY}/{PROJECT}/{run_id}")

    # 1) pull history
    df = run.history(samples=SAMPLES, pandas=True)
    if df.empty:
        print(f"⚠️  {run.name} – empty history")
        continue

    # 2) build 'step' column
    default_steps = pd.Series(np.arange(len(df)), index=df.index)
    df["step"] = (
        df.get("train/global_step")
          .fillna(df.get("_step"))
          .fillna(default_steps)
    ).astype(int)

    # 3) collapse alt keys into single loss columns
    train_cols = [k for k in TRAIN_KEYS if k in df]
    eval_cols  = [k for k in EVAL_KEYS  if k in df]

    df["loss_train"] = df[train_cols].bfill(axis=1).iloc[:, 0] if train_cols else np.nan
    df["loss_eval"]  = df[eval_cols].bfill(axis=1).iloc[:, 0] if eval_cols  else np.nan

    # 4) zero → NaN → ffill
    df.loc[df["loss_train"] == 0, "loss_train"] = np.nan
    df.loc[df["loss_eval"]  == 0, "loss_eval"]  = np.nan
    df[["loss_train", "loss_eval"]] = df[["loss_train", "loss_eval"]].ffill()

    # Normalize loss values into the range 0 to 1 for each column
    for col in ["loss_train", "loss_eval"]:
        if col in df:
            min_val = df[col].min()
            max_val = df[col].max()
            if pd.notnull(min_val) and pd.notnull(max_val) and (max_val - min_val) > 0:
                df[col] = (df[col] - min_val) / (max_val - min_val)

    # 5) append every step of this run
    for _, r in df.iterrows():
        rows.append({
            "model":      run.name,
            "step":       int(r["step"]),
            "loss_eval":  float(r["loss_eval"]),
            "loss_train": float(r["loss_train"])
        })

## import openAI data
df_openai = pd.read_csv("../../open_ai/metadata_fine_tuning_job_ftjob-Cde9cY19khBG4TCRPjeClpAe.csv")[['step', 'train_loss', 'valid_loss']]

# normalize column names
df_openai.rename(columns={
    'train_loss': 'loss_train',
    'valid_loss': 'loss_eval'
}, inplace=True)
# add model name
df_openai['model'] = 'GPT 4o-mini'

# once all runs are done, dump to CSV
per_step_df = pd.DataFrame(rows)

# add openAI data
per_step_df = pd.concat([per_step_df, df_openai], ignore_index=True)

out_path    = "losses_per_step_all_models.csv"
per_step_df.to_csv(out_path, index=False)
print(f"✅ Saved per-step losses to {os.path.abspath(out_path)}")
