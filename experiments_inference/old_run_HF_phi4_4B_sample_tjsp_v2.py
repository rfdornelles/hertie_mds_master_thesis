import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # ajuste se necessário

import time
import json
import datetime
import wandb
import torch
import dotenv
from tqdm import tqdm
from datasets import Dataset
from pandas import read_parquet
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# -----------------------------------------------------------------------------
# 1. CONFIG / SETUP
# -----------------------------------------------------------------------------
dotenv.load_dotenv()
# 

now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment = 'hf_tjsp2024_2025_phi4_14B-ft-inferencee'
folder = f'./{experiment}'
os.makedirs(folder, exist_ok=True)

model_name = 'rfdornelles/master_thesis_phi4_tb_finetune'

# W&B setup
run = wandb.init(
    project="hf_tjsp_drugs_2024_2025_sample",
    name=f"{experiment}_{now}",
    job_type="inference",
    dir=folder,
    config={
        "model": model_name,
        "max_new_tokens": 1000,
        "device_map": "auto",
        # "batch_size": 1,
    }
)
# batch_size = run.config.batch_size

# -----------------------------------------------------------------------------
# 2. LOAD DATA & PROMPT
# -----------------------------------------------------------------------------
df = read_parquet("../data/sample_454_tjsp_drug_cases_2024-2025_v1.parquet")
df = df[["processo", "julgado"]]
ds = Dataset.from_pandas(df)

done = {f.split(".json")[0] for f in os.listdir(folder) if f.endswith(".json")}
ds = ds.filter(lambda x: str(x["processo"]) not in done)

with open('../prompt/prompt_v4.md', 'r') as f:
    system_prompt = f.read().strip()

run.log({"total_cases": len(ds)})

# -----------------------------------------------------------------------------
# 3. LOAD MODEL LOCALLY (Transformers)
# -----------------------------------------------------------------------------
torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # ou `load_in_8bit=True` se preferir
    bnb_4bit_compute_dtype=torch.bfloat16,  # ou torch.float16
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
 
try:
    pipe = pipeline(
        "text-generation",
        model= model,#model_name,
        tokenizer= tokenizer, #model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        # model_kwargs={"attn_implementation": "flash_attention_2"},
        # batch_size=batch_size,
    )
except Exception as e:
    wandb.alert(title="🚨 Model load failed", text=str(e))
    raise

# -----------------------------------------------------------------------------
# 4. HELPERS
# -----------------------------------------------------------------------------
def make_prompt(text):
    return f"{system_prompt}\n\n----- Sentença judicial:\n{text}"

def save_output(pid, content):
    with open(os.path.join(folder, f"{pid}.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": content}, f, ensure_ascii=False)

# -----------------------------------------------------------------------------
# 5. INFERENCE LOOP
# -----------------------------------------------------------------------------
sample_table = wandb.Table(columns=["processo", "latency_s", "snippet"])

for i in tqdm(range(len(ds)), desc="Rodando exemplos"):
    item = ds[0]
    pid = str(item["processo"])
    text = item["julgado"]

    output_path = os.path.join(folder, f"{pid}.json")
    if os.path.exists(output_path):
        continue

    prompt = make_prompt(text)

    try:
        start = time.time()
        output = pipe(
            prompt,
            max_new_tokens=run.config.max_new_tokens,
            do_sample=False,
            return_full_text=False
        )
        elapsed = time.time() - start
        torch.cuda.empty_cache()
    except torch.cuda.OutOfMemoryError as oom:
        wandb.alert(title="🚨 OOM error", text=str(oom))
        raise
    except Exception as e:
        wandb.alert(title="🚨 Inference error", text=str(e))
        raise

    generated_text = output[0]["generated_text"]
    save_output(pid, generated_text)

    wandb.log({"latency_s": elapsed})
    if len(sample_table.data) < 20:
        sample_table.add_data(pid, elapsed, generated_text[:200].replace("\n", " "))


# -----------------------------------------------------------------------------
# 6. LOG TO W&B + FINISH
# -----------------------------------------------------------------------------
run.log({"example_inferences": sample_table})
artifact = wandb.Artifact(f"{experiment}-outputs", type="inference-results")
artifact.add_dir(folder)
run.log_artifact(artifact)
run.finish()

torch.cuda.empty_cache()


