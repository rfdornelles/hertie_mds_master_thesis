import torch
import time
from transformers import AutoTokenizer, Gemma3ForCausalLM
from datasets import load_dataset, load_from_disk
import time 

ckpt = "google/gemma-3-12b-it"
model = Gemma3ForCausalLM.from_pretrained(
    ckpt, 
    torch_dtype=torch.bfloat16, 
    # device_map="cuda",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True)

messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant who is fluent in Shakespeare English"},]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Who are you?"},]
        },
    ],
]



df = load_from_disk("../data/validation").to_pandas()

with open("../prompt/prompt_v4.md", "r") as f:
    prompt = f.read()
    


def run_extraction(text, new_tokens=1000):
    start_time = time.time()
    
    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": prompt},]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": str(text)},]
            },
        ],
    ]
    print(f"Message created: {messages}")
    print(f"Time taken to create message: {time.time() - start_time:.2f} seconds")
    
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)
    print("Input created")
    print(f"Time taken to create input: {time.time() - start_time:.2f} seconds")
    
    input_len = inputs["input_ids"].shape[-1]
    print(f"Input len created: {input_len}")
    print(f"Time taken to get input length: {time.time() - start_time:.2f} seconds")
    
    generation = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
    print("Generation created")
    print(f"Time taken to generate: {time.time() - start_time:.2f} seconds")
    
    generation = generation[0][input_len:]
    print("Decodings")
    
    decoded = tokenizer.decode(generation, skip_special_tokens=True)
    print(decoded)
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")


df["input"][0]
run_extraction(df["input"][0], 2000)