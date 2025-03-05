###### testing finetune llama 3.2 with the validation dataset

# libs
import pandas as pd
from transformers import pipeline
import torch
import datasets

torch.cuda.empty_cache()

# 1. import golden dataset and prompt
df_validation = pd.read_parquet('../data/validation.parquet')

# 2. transform data into json to receive the pairs: (julgado, features)
# all the columns - except julgado - will become a json object

df_data = df_validation.drop(columns=['julgado', 'id']).apply(lambda x: x.to_json(), axis=1)

# add the julgado column
df_data = pd.DataFrame({'julgado': df_validation['julgado'], 'data': df_data})

# read prompt
with open('../prompt/prompt_v2.txt') as f:
    prompt = f.read()

## set the prompt template
def build_prompt(texto_julgado, style = 'off', force_str = True):

  if (style == 'message'):
    prompt_template = [
      {"role": "system", "content": prompt},
      {"role": "user", "content": f"--------- Sentença: ---------\n {texto_julgado}"}
    ]
    
  else:
    prompt_template = f"""TAREFA: Você deverá responder em formato JSON apenas e tão somente.INSTRUÇÕES: {prompt}
    
    ------------------
    
    SENTENÇA PARA PROCESSAR:  {texto_julgado}
    """
  
  if force_str:
    prompt_template = str(prompt_template)
  
  return prompt_template

# 3.1. import the baseline model 
# meta-llama/Llama-3.2-3B-Instruct
model_name = "meta-llama/Llama-3.2-3B-Instruct" #'meta-llama/Llama-3.2-3B'
llm = pipeline('text-generation', 
               model= model_name,
               device_map="auto")

# test
llm("Are you ready to become a MSc on Data Science?")

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
    {"role": "agent", "content": ""},
]

llm(str(messages), max_new_tokens=10000)

test = str(build_prompt(df_data['julgado'][0]))
test2 = str(build_prompt(df_data['julgado'][0], 'OFF'))
llm(test, max_new_tokens=7000, temperature = 0.1)
llm(test2, max_new_tokens=7000, temperature = 0.1)

llm(test, return_full_text=False, max_new_tokens=2000, temperature = 0.1)
# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

# def llama_model(messages, max_length=100, skip_special_tokens=True):
#     inputs = tokenizer(messages, return_tensors="pt")
    
#     generate_ids = model.generate(inputs.inputs, max_length=max_length),
    
#     return tokenizer.batch_decode(generate_ids, skip_special_tokens=skip_special_tokens)

# llama_model(messages)

# 4. start the ft pipeline

# 5. test the output

## finetune with huggingface