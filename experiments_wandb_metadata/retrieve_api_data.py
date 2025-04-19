## Retrieve data from Weights and Biases API

from dotenv import load_dotenv
import os
import wandb
import json
from json import JSONDecodeError
import pandas as pd

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
