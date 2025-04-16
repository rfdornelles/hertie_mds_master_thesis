import torch
from transformers import AutoTokenizer, Gemma3ForCausalLM
from datasets import load_from_disk, concatenate_datasets
import os
import tqdm

# clean cuda just in case
torch.cuda.empty_cache()
# CUDA_VISIBLE_DEVICES="0,2,3"
# definitions
experiment = 'experiment_gemma3_27b_it_baseline_v2'
folder = f'experiments2/{experiment}'
model_name = "google/gemma-3-27b-it"

os.makedirs(folder, exist_ok=True)

# load datasets
validation = load_from_disk('../data/validation')
test = load_from_disk('../data/test')

# merge both
datasets = concatenate_datasets([validation, test]).to_pandas()

# load prompt
with open('../prompt/prompt_v4.md', 'r') as f:
    prompt = f.read()

## load model

model = Gemma3ForCausalLM.from_pretrained(
    model_name, 
    torch_dtype=torch.bfloat16, 
    # load_in_8bit=True,
    # device_map="cuda",
    device_map="auto",
    attn_implementation = 'eager'
)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

## load model
print("Model and datasets loaded!!")

## function to build the prompt
def build_prompt(input):
  messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": str(input)},]
            },
        ],
    ]
    
  return messages


## function to iterate
def run_model(processo, new_tokens=1000, overwrite = False):
  
  # clean: remove everything that's not numbers
  processo = str(processo).replace(r'[^0-9]', '')
  
  # check if the file already exists
  file = f'{folder}/{processo}.json'
  
  if os.path.exists(file) and not overwrite:
    print(f"File {file} already exists. Skipping...")
    return

  print(f"  Processing {processo}...")
  # retrieve the content of the sentença
  sentenca = datasets[datasets['processo'] == processo]['input'].values[0]
  
  # build the prompt
  message = build_prompt(sentenca)
  
  # run the model
  try:
    print(f"  Applying chat template for {processo}")
    inputs = tokenizer.apply_chat_template(
        message, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device)
    
    input_len = inputs["input_ids"].shape[-1]
    
    print(f"  Generating for {processo}")
    generation = model.generate(**inputs, 
                                max_new_tokens=new_tokens, 
                                do_sample=False,
                                temperature=None,
                                top_p=None,
                                top_k=None)
  
    print(f"  Decoding for {processo}")
    # only keep the new tokens
    generation = generation[0][input_len:]
  
    print("  Decoding the generation...")
    decoded = tokenizer.decode(generation, skip_special_tokens=True)
    
  except Exception as e:
    print(f"Error processing {processo}: {e}")
    return
    
    
  # save the output as json
  print(f"  Saving the output for {processo}...")
  try:
    with open(file, 'w') as f:  
      f.write(decoded)
      return True
    
  except Exception as e:
    print(f"Error writing file {file}: {e}")
    return  
  
  
# test    
#run_model('00316684320178260050', overwrite = True)

# with open(f'{folder}/00316684320178260050.json', 'r') as f:
#     output = f.read()
 
## iterate
print("Starting the iteration...")

for processo in tqdm.tqdm(datasets['processo'].unique()):
  try:
    run_model(processo)
  except Exception as e:
    print(f"Error processing {processo}: {e}")
    continue
  

torch.cuda.empty_cache()    
print("All processes finished -- cache cleaned.")   

    
