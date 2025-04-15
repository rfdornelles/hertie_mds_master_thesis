### objective: run baseline llama on train and validation datasets

### references:
# https://medium.com/@alejandro7899871776/structure-output-with-llama-from-scratch-39c487b6be81
# https://www.datacamp.com/tutorial/fine-tuning-llama-3-2
# https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/
# https://python.langchain.com/docs/how_to/output_parser_json/

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2" # set the GPU devices to use
### imports
from transformers import pipeline
from datasets import load_from_disk, concatenate_datasets
import torch

import tqdm

# clean cuda just in case
torch.cuda.empty_cache()

# definitions
experiment = 'experiment_finetuned_unsloth_Phi-4'
folder = f'experiments2/{experiment}/'
os.makedirs(folder, exist_ok=True)


# load datasets
validation = load_from_disk('../data/validation')
test = load_from_disk('../data/test')

# merge both
datasets = concatenate_datasets([validation, test]).to_pandas()

# load prompt
with open('../prompt/prompt_v4.md', 'r') as f:
    prompt = f.read()

# load model
model = 'ft_hf_lora_Phi-4_v1_2025-04-15_14-00-56'

pipe = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print("Model and datasets loaded!!")

## function to build the prompt
def build_prompt(input):

  messages = [
            {'role': 'user', 'content': prompt + f"----- Sentença judicial:     {input}"}
        ]
  
  return messages

## function to iterate
def run_model(processo, overwrite = False):
  
  # clean: remove everything that's not numbers
  processo = str(processo).replace(r'[^0-9]', '')
  
  # check if the file already exists
  file = f'{folder}/{processo}.json'
  if os.path.exists(file) and not overwrite:
    print(f"File {file} already exists. Skipping...")
    return
  
  # retrieve the content of the sentença
  sentenca = datasets[datasets['processo'] == processo]['input'].values[0]
  
  # build the prompt
  message = build_prompt(sentenca)
  
  # run the model
  try:
    outputs = pipe(
        message,
        max_new_tokens=1000,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
    )
    
  except Exception as e:
    print(f"Error processing {processo}: {e}")
    return
  
  # save the output as json
  output = outputs[0]["generated_text"][-1]['content']
  try:
    with open(file, 'w') as f:  
      f.write(output)
      return True
  except Exception as e:
    print(f"Error writing file {file}: {e}")
    return
  
  
# test the process
# run_model('00316684320178260050', overwrite = True)

# with open(f'{folder}/00316684320178260050.json', 'r') as f:
#     output = f.read()
    
# parser.parse(output)

## iterate
print("Starting the iteration...")

for processo in tqdm.tqdm(datasets['processo']):
  try:
    run_model(processo)
  except Exception as e:
    print(f"Error processing {processo}: {e}")
    continue

torch.cuda.empty_cache()
print("All processes finished. Cache cleaned!")

# unload the model
del pipe
del model
torch.cuda.empty_cache()
print("Model unloaded. Cache cleaned!")
exit(0)