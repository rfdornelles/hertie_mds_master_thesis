import pandas as pd
import tiktoken
import os


df = pd.read_parquet("../data/sample_454_tjsp_drug_cases_2024-2025_v1.parquet")

## length of the tokens using the same as GPT-4 (via tiktoken)

encoding = tiktoken.encoding_for_model("gpt-4")

def tokenize_text(text):
  tokens = encoding.encode(text)
  return len(tokens)

df['tokens'] = df['julgado'].apply(tokenize_text)

# total tokens
total_tokens = df['tokens'].sum()
print(f"Total tokens: {total_tokens}")

# tokenize the prompt
with open("../prompt/prompt_v4.md", "r") as f:
  prompt = f.read()
  
prompt_tokens = tokenize_text(prompt)
print(f"Prompt tokens: {prompt_tokens}")

# calculate the number of tokens for each row
df['tokens_total'] = df['tokens'].apply(lambda x: x + prompt_tokens)

total_tokens_final = df['tokens_total'].sum()
print(f"Total tokens + prompt: {total_tokens_final}")

### output tokens

# read the files in the experiment folder
files = os.listdir(f"../tjsp_experiment/tjsp_2024_2025_sample_454")
  
# only if the file ends with .json
files = [f for f in files if f.endswith(".json")]

output_tokens = 0
# read and sum the tokens
for file in files:
  with open(f"../tjsp_experiment/tjsp_2024_2025_sample_454/{file}", "r") as f:
    content = f.read()
    output_tokens += tokenize_text(content)
    # print(f"Output tokens for {file}: {output_tokens}")
print(f"Total output tokens: {output_tokens}")

## calculate the cost

### gpt-4o-mini (batch)
# input: 0.075 / million
# output: 0.30 / million

## ft
# input: 0.15 / million
# output: 0.60 / million

input_tokens = total_tokens_final
output_tokens = output_tokens
cost_input = 0.13 # euro in 26/04/2025
cost_output = 0.53 # euro in 26/04/2025 

cost = (input_tokens/1000000) * cost_input + (output_tokens/1000000) * cost_output
print(f"Total cost (GPT 4o-mini FT batch mode): {cost} EUR")

## eletricity berlin 
# https://euenergy.live/electricity-prices/germany/berlin


