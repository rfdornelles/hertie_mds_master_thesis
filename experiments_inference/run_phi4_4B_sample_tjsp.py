import time
import os
import torch
import tqdm
import wandb
import datetime
import json
from pandas import read_parquet
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from datasets import Dataset
import dotenv

CUDA_VISIBLE_DEVICES = '2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES

dotenv.load_dotenv()
# -----------------------------------------------------------------------------
# 1. CONFIG / SETUP
# -----------------------------------------------------------------------------
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment = 'hf_tjsp2024_2025_phi4_14B-ft-inferencee'
folder     = f'../{experiment.split("/")[0]}'
os.makedirs(folder, exist_ok=True)
model_name = 'rfdornelles/master_thesis_phi4_tb_finetune'


# Initialize W&B run
run = wandb.init(
    project="tjsp_drugs_2024_sample",
    name=f"{experiment}_{now}",
    job_type="inference",
    config={
        "model": model_name,
        "max_new_tokens": 1000,
        "device_map": "auto",
        "batch_size": 1,
        "cuda_visible_devices": CUDA_VISIBLE_DEVICES,
    }
)

# batch size
batch_size = run.config.batch_size
# -----------------------------------------------------------------------------
# 2. LOAD DATA & MODEL
# -----------------------------------------------------------------------------
# load your merged dataset


datasets = read_parquet("../data/sample_454_tjsp_drug_cases_2024-2025_v1.parquet")#"../data/sample400_tjsp_drug_cases_2024_v2.parquet")
datasets = datasets[["processo", "julgado"]]
datasets = Dataset.from_pandas(datasets)

# remove files already generated
processed = {
    fname.replace(".json", "")
    for fname in os.listdir(folder)
    if fname.endswith(".json")
}

datasets = datasets.filter(lambda ex: ex["processo"] not in processed)



# load your system prompt
with open('../prompt/prompt_v4.md', 'r') as f:
    prompt = f.read()

# instantiate the HF text-generation pipeline

torch.cuda.empty_cache()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # use 4-bit
    bnb_4bit_quant_type="nf4",            # normal-float4 (nf4) quantization
    bnb_4bit_compute_dtype=torch.bfloat16,# compute in bfloat16 for better perf
    bnb_4bit_use_double_quant=True        # two-stage (double) quantization for higher accuracy
)


try:
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# 3) Load the tokenizer as usual
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
    )

except Exception as e:
    wandb.alert(
        title="🚨 Model loading error",
        text=f"Error loading model {model_name}: {e}"
    )
    raise

wandb.log({"total_cases": len(datasets), "stratify_by": "group"})

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def build_prompt(text: str):
    return [
            {'role': 'user', 'content': prompt + f"----- Sentença judicial:     {text}"}
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
        outputs = pipe(messages, 
                       max_new_tokens=1000, 
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


def infer_batch(batch):
    
    texts = batch["julgado"]
    # build list of chat messages
    messages = [
        [
            {"role": "system", "content": prompt},
            {"role": "user",   "content": txt}
        ]
        for txt in texts
    ]

    t0 = time.time()
    outputs = pipe(
        messages,
        max_new_tokens=run.config.max_new_tokens,
        do_sample=False,
        top_p = None,
        temperature = None
    )
    
    elapsed = time.time() - t0

    # evenly assign per‑sample latency
    per_sample_latency = elapsed / len(texts)

    # parse generated text
    gen = [
        out[0]["generated_text"][-1]["content"]
        for out in outputs
    ]

    return {
        "generated": gen,
        "latency_s": [per_sample_latency] * len(texts)
    }

# -----------------------------------------------------------------------------
# 4. RUN INFERENCE
# -----------------------------------------------------------------------------
sample_table = wandb.Table(columns=["processo", "latency_s", "snippet"])
# for proc in tqdm.tqdm(datasets):
    
# # res = run_single(proc['processo'])
# result = datasets.map(
#     infer_batch,
#     batched=True,
# # batch_size= 4, #run.config.batch_size,
# # # disable_tqdm=True    # <- silence this one call only
# )
torch.cuda.empty_cache()

    # if res:
    #     # log every inference as a point in a latency chart
    #     wandb.log({"latency_s": res["latency_s"]})

    #     # capture a handful of examples in a W&B table
    #     if len(sample_table.data) < 20:
    #         snippet = res["output"][:200].replace("\n", " ")
    #         sample_table.add_data(res["processo"], res["latency_s"], snippet)

try:
    # ─── Inference Loop ───────────────────────────────────────────────────────
    for i in tqdm.tqdm(range(0, len(datasets), batch_size), desc="Batches"):
        batch = datasets.select(range(i, min(i + batch_size, len(datasets))))
        ids    = batch["processo"]
        texts  = batch["julgado"]

        # filter out those already done
        work_items = [
            (pid, txt)
            for pid, txt in zip(ids, texts)
            if not os.path.exists(os.path.join(folder, f"{pid}.json"))
        ]
        if not work_items:
            continue  # skip entire batch

        pids, inputs = zip(*work_items)

        # build prompts only for the ones we need
        messages = [
            [
                {'role': 'user', 'content': prompt + f"----- Sentença judicial:     {txt}"}
            ]
            for txt in inputs
        ]

        # run model once per batch
        start   = time.time()
        outputs = pipe(messages, 
                    max_new_tokens=1000, 
                    do_sample=False, 
                    temperature = None, 
                    top_p = None)
        
        torch.cuda.empty_cache()
        elapsed = time.time() - start

        # extract and write each JSON
        for pid, out in zip(pids, outputs):
            
            gen_text = out[0]["generated_text"][-1]["content"]
            out_path = os.path.join(folder, f"{pid}.json")
            with open(out_path, 'w', encoding='utf-8') as f:
                # dump only the content string as valid JSON
                f.write(gen_text)

except torch.cuda.OutOfMemoryError as oom:
   wandb.alert(
        title="🚨 OOM during inference",
        text=f"OOM GPU on batch {i}: {oom}"
    )
   raise

except Exception as e:
    wandb.alert(
        title="🚨 Run crashed",
        text=f"Error on batch {i} {type(e).__name__}: {e}"
    )
    raise


# push the sample outputs to W&B
run.log({"example_inferences": sample_table})

# 5. PACKAGE ALL JSONS AS AN ARTIFACT
artifact = wandb.Artifact(f"{experiment}-outputs", type="inference-results")
artifact.add_dir(folder)  
run.log_artifact(artifact)

# 6. FINISH
run.finish()

torch.cuda.empty_cache()