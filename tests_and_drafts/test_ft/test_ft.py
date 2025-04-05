###### testing finetune llama 3.2 with the validation dataset

# libs
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline
)
import torch
from datasets import Dataset
from transformers import TrainerCallback


torch.cuda.empty_cache()
# 1. import golden dataset and prompt
df_validation = pd.read_parquet('validation.parquet')

# 2. transform data into json to receive the pairs: (julgado, features)
# all the columns - except julgado - will become a json object

df_data = df_validation.drop(columns=['julgado', 'id']).apply(lambda x: x.to_json(), axis=1)

# add the julgado column
df_data = pd.DataFrame({'julgado': df_validation['julgado'], 'data': df_data})

# read prompt
with open('prompt_v2.txt') as f:
    global_prompt = f.read()
  

######## finetune with huggingface

## create dataset
dataset_train = Dataset.from_pandas(df_data[:30])
dataset_test = Dataset.from_pandas(df_data[30:])

## init pipeline
# meta-llama/Llama-3.2-3B-Instruct
model_name = "meta-llama/Llama-3.2-1B-Instruct" #meta-llama/Llama-3.2-3B-Instruct" #'meta-llama/Llama-3.2-3B'
nickname = model_name.split('/')[1]


model = AutoModelForCausalLM.from_pretrained(model_name)
context_window = int(model.config.max_position_embeddings/18)
print("Janela de contexto (max tokens):", context_window)


tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
    
def tokenize_function(examples, max_length= context_window):
    return tokenizer(
      examples['julgado'], 
      padding="max_length", 
      truncation=True,
      max_length=max_length,
    )


def preprocess_function(examples, max_length=context_window):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for julgado, data in zip(examples["julgado"], examples["data"]):
        prompt_template = f"""TAREFA: Você deverá responder em formato JSON apenas. 
        INSTRUÇÕES: {global_prompt} ------------------ 
        SENTENÇA: {julgado} 
        RESPOSTA:"""
        resposta = data

        # Tokenizar prompt + resposta JUNTOS
        full_text = prompt_template + resposta
        tokenized = tokenizer(
            full_text, 
            max_length=max_length,
            truncation=True,
            padding="max_length",  # Garantir padding fixo
            return_tensors="pt"
        )
        
        # Tokenizar APENAS o prompt
        prompt_only = tokenizer(
            prompt_template, 
            max_length=max_length,
            truncation=True,
            padding="max_length",  # Mesmo padding
            return_tensors="pt"
        )
        
        # Calcular comprimento real do prompt
        prompt_length = (prompt_only["attention_mask"] == 1).sum().item()
        
        # Garantir que o prompt não ultrapasse o limite
        prompt_length = min(prompt_length, max_length)
        
        # Criar labels mesmo se a resposta for truncada
        labels = tokenized["input_ids"].clone()
        labels[:, :prompt_length] = -100
        labels[tokenized["attention_mask"] == 0] = -100
        
        input_ids_list.append(tokenized["input_ids"].squeeze())
        attention_mask_list.append(tokenized["attention_mask"].squeeze())
        labels_list.append(labels.squeeze())
    
    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }


model.config.use_cache = False  # Disable caching for gradient checkpointing compatibility
model.gradient_checkpointing_enable()


###

print("Init preprocess")

# Antes do treinamento, verificar exemplos problemáticos
for i in range(len(df_data)):
    if not isinstance(df_data.iloc[i]['data'], str) or len(df_data.iloc[i]['data']) < 10:
        print(f"Exemplo {i} com dados inválidos: {df_data.iloc[i]}")
        df_data = df_data.drop(i)


tokenized_train_datasets = dataset_train.map(preprocess_function, batched=True).with_format("torch")
tokenized_test_datasets = dataset_test.map(preprocess_function, batched=True).with_format("torch")

# Após o pré-processamento
for i in range(len(tokenized_train_datasets)):
    input_ids = tokenized_train_datasets[i]["input_ids"]
    labels = tokenized_train_datasets[i]["labels"]
    
    # Verificar se há tokens não mascarados
    if (labels == -100).all():
        raise ValueError(f"Exemplo {i}: Todos os labels estão mascarados!")
    
    # Verificar NaN/Inf
    if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
        raise ValueError(f"Exemplo {i}: Input_ids contém valores inválidos!")
      
      
print("Finish preprocess")

class GradientMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            grads = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            if len(grads) == 0:
                print("Gradientes não encontrados!")
            else:
                print(f"Gradiente médio: {sum(grads)/len(grads)}")


class CustomCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if 'loss' in logs:
            if torch.isnan(torch.tensor(logs['loss'])).any():
                print("Loss contém NaN! Interrompendo treinamento.")
                raise ValueError("NaN detected in loss")

# training args
training_args = TrainingArguments(
  output_dir=f"./results-{nickname}",
  eval_strategy='steps',
  learning_rate=1e-5,
  per_device_train_batch_size=1,
  per_device_eval_batch_size=1,
  gradient_accumulation_steps=4,
  num_train_epochs=3,
  weight_decay=0.1,
  warmup_ratio=0.1,
  save_total_limit=2,
  fp16=True,
  bf16=False,    # Habilite bf16
  optim="adamw_torch",
  max_grad_norm=1.0,
  logging_dir=f"./logs-{nickname}",
  logging_steps=10,
  report_to="none",
  gradient_checkpointing=True,
)

# trainer
trainer = Trainer(
  model=model,
  args=training_args,
  train_dataset=tokenized_train_datasets,
  eval_dataset=tokenized_test_datasets,
  processing_class = tokenizer,
  callbacks=[CustomCallback(), GradientMonitorCallback()],
)

print("Init training")
# run
trainer.train()

print("Finish training")