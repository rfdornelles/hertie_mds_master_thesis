### objective: run baseline llama on train and validation datasets

### references:
# https://medium.com/@alejandro7899871776/structure-output-with-llama-from-scratch-39c487b6be81
# https://www.datacamp.com/tutorial/fine-tuning-llama-3-2
# https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/
# https://python.langchain.com/docs/how_to/output_parser_json/

### imports
from transformers import pipeline
from datasets import load_from_disk, concatenate_datasets
import torch
import os
import tqdm

# clean cuda just in case
torch.cuda.empty_cache()

# definitions
experiment = 'experiment_llama_3_2_baseline'
folder = f'experiments/{experiment}/'
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
model = 'meta-llama/Llama-3.2-3B-Instruct'

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
    {"role": "system", "content": prompt},
    {"role": "user", "content": input},
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
        do_sample=False
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

for processo in tqdm.tqdm(datasets['processo'].unique()):
  try:
    run_model(processo, overwrite = True)
  except Exception as e:
    print(f"Error processing {processo}: {e}")
    continue
  finally:
    torch.cuda.empty_cache()
    #print(f"Process {processo} finished.")
    
print("All processes finished.")