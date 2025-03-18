#%% Instalação de dependências (se necessário)
#!pip install -q -U transformers datasets accelerate peft bitsandbytes flash-attn

#%% Imports
import torch
import os
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_from_disk
from accelerate import Accelerator 

#%% Configurações globais
MODEL_NAME = "TucanoBR/Tucano-2b4-Instruct"   # Modelo base
CONTEXT_TARGET = 16384                       # 16k tokens
NUM_EPOCHS = 3

# Ajustes LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

# Parâmetros de treino
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUM_STEPS = 4
LEARNING_RATE = 2e-5

#%% 1. Carregar modelo com Positional Interpolation
def load_model_with_interpolation():
    rope_scaling = {"type": "linear", "factor": 4.0}
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # Carregar modelo com mapeamento explícito de dispositivo
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        rope_scaling=rope_scaling,
        # device_map={"": f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"}  # Corrige o device mapping
    )
    
    return model
#%% 2. Setup do LoRA
def setup_lora(model):
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
        modules_to_save=["embed_tokens", "norm"]  # Mantém essas camadas em float32
    )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    
    # Forçar conversão de parâmetros LoRA para float32
    for param in model.parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    
    model.print_trainable_parameters()
    return model

#%% 3. Carregar e preparar o dataset
def load_custom_dataset():
    """Carrega o dataset que foi salvo previamente em formato HuggingFace.
       O dataset está na pasta 'data/test_lora'.
    """
    dataset = load_from_disk("data/test_lora")
    return dataset

def chunk_text(text, tokenizer, chunk_size=CONTEXT_TARGET, overlap=256):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk = tokens[start:end]
        chunk_text_decoded = tokenizer.decode(chunk)
        chunks.append(chunk_text_decoded)
        start += (chunk_size - overlap)
    return chunks

def format_as_instruct(text):
    return f"""### Instrução:
Leia o texto a seguir e responda resumidamente:

### Texto:
{text}

### Resposta:
"""

def prepare_dataset():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    raw_dataset = load_custom_dataset()

    def process_fn(examples):
        # Ajuste o nome da coluna conforme seu dataset (ex.: "julgado", "texto", etc.)
        all_texts = examples["julgado"]
        
        output_texts = []
        for text in all_texts:
            chunks = chunk_text(text, tokenizer, chunk_size=CONTEXT_TARGET, overlap=256)
            for c in chunks:
                instruct_text = format_as_instruct(c)
                output_texts.append(instruct_text)
        
        return {"input_text": output_texts}
    
    ds_processed = raw_dataset.map(
        process_fn,
        batched=True,
        remove_columns=raw_dataset.column_names
    )
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["input_text"],
            truncation=True,
            max_length=CONTEXT_TARGET,
            padding="max_length"
        )
    
    ds_tokenized = ds_processed.map(tokenize_fn, batched=True)
    ds_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    return ds_tokenized

def custom_auto_wrap_policy(module, recurse, unwrapped_params):
    # Skip modules with non-float parameters (quantized layers)
    for p in module.parameters(recurse=False):
        if not p.dtype.is_floating_point:
            return False
    
    # Check submodules recursively if needed
    if recurse:
        return True
    
    # Only wrap modules larger than 100M parameters
    param_count = sum(p.numel() for p in module.parameters(recurse=False))
    return param_count > 100_000_000

#%% 4. Função de treinamento principal
def train():
    # Inicialização do Accelerator
    accelerator = Accelerator()
    
    # Carregar modelo e tokenizer
    model = load_model_with_interpolation()
    model = setup_lora(model)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Preparar dataset
    dataset = prepare_dataset()
    
    # Configuração do DeepSpeed
    ds_config = {
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": LEARNING_RATE,
                "betas": [0.9, 0.999],
                "weight_decay": 0.001
            }
        },
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto"
        },
        "gradient_accumulation_steps": GRADIENT_ACCUM_STEPS,
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "wall_clock_breakdown": False
    }

    training_args = TrainingArguments(
        output_dir="./tucano-16k",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        optim="paged_adamw_32bit",
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none",
        deepspeed=ds_config  # Usar DeepSpeed ao invés de FSDP
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )

    # Verificação final de tipos de dados
    for name, param in model.named_parameters():
        if not param.dtype.is_floating_point:
            print(f"Parâmetro não-float detectado: {name} - {param.dtype}")
            param.data = param.data.to(torch.float32)  # Conversão forçada

    trainer.train()
    trainer.save_model("./tucano-16k-final")

#%% 5. Execução principal
if __name__ == "__main__":
    train()
