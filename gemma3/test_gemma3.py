# $ pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# from transformers import pipeline
# import torch

# pipe = pipeline(
#     "text-generation",#"image-text-to-text",
#     model="google/gemma-3-12b-it",
#     device="cuda",
#     torch_dtype=torch.bfloat16
# )

# messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
#             {"type": "text", "text": "Describe me carefully the picture?"}
#         ]
#     }
# ]

# output = pipe(text=messages, max_new_tokens=200)
# print(output[0]["generated_text"][-1]["content"])
# # Okay, let's take a look! 
# # Based on the image, the animal on the candy is a **turtle**. 
# # You can see the shell shape and the head and legs.

# def describe_imgs(url):
    
#     messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant."}]
#     },
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": url},
#             {"type": "text", "text": "Describe me carefully the picture?"}
#         ]
#     }
# ]
    
#     output = pipe(text=messages, max_new_tokens=200)
#     print(output[0]["generated_text"][-1]["content"])
    
   
# describe_imgs("https://www.mercadoeeventos.com.br/wp-content/uploads/2022/10/Embratur-Brasil-ultrapassa-marca-de-1-milhao-de-turistas-estrangeiros-recebidos-pela-primeira-vez-desde-2020.png") 

# ### read the dataset
# from datasets import load_dataset, load_from_disk

# data = load_from_disk("test_lora")

# def summmarise(text):
    
#     messages = [
#     {
#         "role": "system",
#         "content": [{"type": "text", "text": "You are a helpful assistant. Summarise the judicial sentence."}]
#     },
#     {
#         "role": "user",
#         "content": [{"type": "text", "text": str(text)}]
#     }
#     ]
    
#     output = pipe(text=messages, max_new_tokens=200)
#     print(output[0]["generated_text"][-1]["content"])
    
    
# summmarise("1234 dlete")
#---
import torch
import time
from transformers import AutoTokenizer, Gemma3ForCausalLM

ckpt = "google/gemma-3-4b-it"
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

# inputs = tokenizer.apply_chat_template(
#     messages, add_generation_prompt=True, tokenize=True,
#     return_dict=True, return_tensors="pt"
# ).to(model.device)

# input_len = inputs["input_ids"].shape[-1]

# generation = model.generate(**inputs, max_new_tokens=100, do_sample=False)
# generation = generation[0][input_len:]

# decoded = tokenizer.decode(generation, skip_special_tokens=True)
# print(decoded)

# def summarize(text, new_tokens=200):
#     start_time = time.time()
    
#     messages = [
#         [
#             {
#                 "role": "system",
#                 "content": [{"type": "text", "text": "You are a helpful assistant who is fluent in Portuguese. Summarise the judicial sentence, in Brazilian Portuguese."},]
#             },
#             {
#                 "role": "user",
#                 "content": [{"type": "text", "text": str(text)},]
#             },
#         ],
#     ]
#     print(f"Message created: {messages}")
#     print(f"Time taken to create message: {time.time() - start_time:.2f} seconds")
    
#     inputs = tokenizer.apply_chat_template(
#         messages, add_generation_prompt=True, tokenize=True,
#         return_dict=True, return_tensors="pt"
#     ).to(model.device)
#     print("Input created")
#     print(f"Time taken to create input: {time.time() - start_time:.2f} seconds")
    
#     input_len = inputs["input_ids"].shape[-1]
#     print(f"Input len created: {input_len}")
#     print(f"Time taken to get input length: {time.time() - start_time:.2f} seconds")
    
#     generation = model.generate(**inputs, max_new_tokens=new_tokens, do_sample=False)
#     print("Generation created")
#     print(f"Time taken to generate: {time.time() - start_time:.2f} seconds")
    
#     generation = generation[0][input_len:]
#     print("Decodings")
    
#     decoded = tokenizer.decode(generation, skip_special_tokens=True)
#     print(decoded)
#     print(f"Total time taken: {time.time() - start_time:.2f} seconds")

# from datasets import load_dataset, load_from_disk
# import time

# data = load_from_disk("test_lora")

# # test 1
# test = data["julgado"][33]
# len(test)
# # summarize(test, 2000)

## read dataset
import pandas as pd

df = pd.read_parquet("../data/validation.parquet")

with open("prompt_v2.txt", "r") as f:
    prompt = f.read()
    


def run_extraction(text, new_tokens=500):
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

from datasets import load_dataset, load_from_disk
import time 

df["julgado"][0]
run_extraction(df["julgado"][0], 2000)