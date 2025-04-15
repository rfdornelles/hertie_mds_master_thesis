# ref: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb#scrollTo=UBuVBy_sU2Yw
# ref: https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3" # Set the visible devices

from unsloth import FastLanguageModel
import torch
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats
from unsloth.chat_templates import train_on_responses_only

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

from transformers import EarlyStoppingCallback
from datasets import load_from_disk
import torch
import dotenv
import wandb

import datetime
## definitions

ckpt = "ricdomolm/lawma-8b" #"unsloth/gemma-3-27b-it-GGUF" #"unsloth/gemma-3-27b-it-unsloth-bnb-4bit" #"google/gemma-3-4b-it"
max_seq_length = 2048*9
num_train_epochs = 10
full_finetuning = False
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = f"ft_unsloth_{'full_finetunning' if full_finetuning else ''}_{ckpt.split('/')[1]}_v1_{now}"

# Load environment variables from .env file
dotenv.load_dotenv()
# Initialize wandb
wandb.login()

torch.cuda.empty_cache()


model, tokenizer = FastLanguageModel.from_pretrained(
  model_name = ckpt,
  attn_implementation='eager',
  dtype=None,
  max_seq_length = max_seq_length, # Choose any for long context!
  load_in_4bit = True,  # 4 bit quantization to reduce memory
  full_finetuning = full_finetuning, # [NEW!] We have full finetuning now!
  # token = "hf_...", # use one if using gated models
  # device_map = "auto",  # Map model layers to CUDA devices 1, 2, and 3 (via CUDA_VISIBLE_DEVICES)
#   use_gradient_checkpointing=
)

## peft
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.2",
)

print("Model and tokenizer loaded!!")


train = load_from_disk("../data/train")
train = standardize_data_formats(train)

test = load_from_disk("../data/validation")
test = standardize_data_formats(test)

with open("../prompt/prompt_v4.md", "r") as f:
    prompt = f.read()

def build_template(batch):
    outputs = []
    # Itera sobre cada par input/output do batch
    for inp, out in zip(batch["input"], batch["output"]):
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f"----- Sentença judicial:     {inp}"},
            {'role': 'assistant', 'content': out}
        ]
        # Aplica o template para cada exemplo e adiciona o resultado na lista
        outputs.append(tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False))
    return {"text": outputs}

train = train.map(build_template, batched=True)
test = test.map(build_template, batched=True)

# # get the largest sequence of tokens
# train_max_length = max([len(tokenizer(x)['input_ids']) for x in train["text"]])
# test_max_length =  max([len(tokenizer(x)['input_ids']) for x in test["text"]])

# identify the largest example and get it 'processo'
# test.max_length = max([len(tokenizer(x)['input_ids']) for x in test["text"]])
# test

print("Train and test datasets loaded!!")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train,
    eval_dataset = test, 
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 1,
    # packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 1,
        per_device_eval_batch_size= 1, 
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = num_train_epochs, # Set this for 1 full training run.
        # max_steps = 60,
        learning_rate = 2e-5,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = f"outputs_{experiment_name}",
        run_name=f"{experiment_name}",
        report_to = "wandb", # Use this for WandB etc
        load_best_model_at_end=True,
        save_strategy = "steps",
        save_steps = 10,  # [NEW!] Save model every 10,000 steps
        save_total_limit=3,
        metric_for_best_model= "eval_loss",
        eval_strategy = "steps",
        eval_steps=10,
        # greater_is_better=False,
    ),
    callbacks=[
      EarlyStoppingCallback(early_stopping_patience=5),
    ],
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>system<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
) 

print("Trainer loaded!!")


# Set the wandb project name
wandb.init(name=f"{experiment_name}")

print("Starting training....")
# with torch.autograd.detect_anomaly():
#     trainer.train()
trainer_stats = trainer.train()
wandb.finish()


print("Training finished!!")

model.save_pretrained(f"{experiment_name}_model")  # Local saving
tokenizer.save_pretrained(f"{experiment_name}_tokenizer")  # Local saving

print("Saving model and tokenizer....")
model.save_pretrained_merged(experiment_name, tokenizer)
print("Model and tokenizer saved!!")

