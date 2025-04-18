import time
import os
import torch
import tqdm
import wandb
import datetime
from pandas import read_parquet
from transformers import pipeline
from datasets import Dataset
import dotenv

dotenv.load_dotenv()
# -----------------------------------------------------------------------------
# 1. CONFIG / SETUP
# -----------------------------------------------------------------------------
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment = f'tjsp2024_llama3.2-3B-ft-inference_{now}'
folder     = f'../experiments_inference/{experiment}/'
os.makedirs(folder, exist_ok=True)
model_name = 'master_thesis_llama_3.2_3b_ft'

# Initialize W&B run
run = wandb.init(
    project="tjsp_drugs_2024_sample",
    name=experiment,
    job_type="inference",
    config={
        "model": model_name,
        "max_new_tokens": 1000,
        "device_map": "auto",
    }
)

# -----------------------------------------------------------------------------
# 2. LOAD DATA & MODEL
# -----------------------------------------------------------------------------
# load your merged dataset


datasets = read_parquet("../data/sample400_tjsp_drug_cases_2024.parquet")
datasets = datasets[["processo", "julgado"]]
datasets = Dataset.from_pandas(datasets)

# load your system prompt
with open('../prompt/prompt_v4.md', 'r') as f:
    prompt = f.read()

# instantiate the HF text-generation pipeline

torch.cuda.empty_cache()
pipe = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

wandb.log({"total_cases": len(datasets), "stratify_by": "group"})

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def build_prompt(text: str):
    return [
        {"role": "system", "content": prompt},
        {"role": "user",   "content": text}
    ]

def run_single(processo: str, overwrite=False):
    processo = "".join(filter(str.isdigit, str(processo)))
    out_path  = os.path.join(folder, f"{processo}.json")
    if os.path.exists(out_path) and not overwrite:
        return None

    text = datasets.filter(lambda x: x['processo'] == processo)['julgado'][0]
    messages = build_prompt(text)

    start = time.time()
    try:
        outputs = pipe(messages, max_new_tokens=1000, 
                       do_sample=False, 
                       top_p = None,
                       temperature = None)
        
        latency = time.time() - start

        content = outputs[0]["generated_text"][-1]["content"]
        with open(out_path, 'w') as f:
            f.write(content)

        return {"processo": processo, "latency_s": latency, "output": content}
    except Exception as e:
        wandb.log({"errors": 1})
        print(f"[ERROR] {processo} → {e}")
        return None

# -----------------------------------------------------------------------------
# 4. RUN INFERENCE
# -----------------------------------------------------------------------------
sample_table = wandb.Table(columns=["processo", "latency_s", "snippet"])
for proc in tqdm.tqdm(datasets):
    
    res = run_single(proc['processo'])
    torch.cuda.empty_cache()

    if res:
        # log every inference as a point in a latency chart
        wandb.log({"latency_s": res["latency_s"]})

        # capture a handful of examples in a W&B table
        if len(sample_table.data) < 20:
            snippet = res["output"][:200].replace("\n", " ")
            sample_table.add_data(res["processo"], res["latency_s"], snippet)

# push the sample outputs to W&B
run.log({"example_inferences": sample_table})

# 5. PACKAGE ALL JSONS AS AN ARTIFACT
artifact = wandb.Artifact(f"{experiment}-outputs", type="inference-results")
artifact.add_dir(folder)  
run.log_artifact(artifact)

# 6. FINISH
run.finish()
