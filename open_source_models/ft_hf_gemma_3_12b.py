import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"  # Set visible devices

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from datasets import load_from_disk
import torch
import dotenv
import wandb
import datetime
from jinja2 import Template

## Configuration
ckpt = "google/gemma-3-12b-it"
max_seq_length = 2048 * 9
num_train_epochs = 10
full_finetuning = False
now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_name = f"ft_hf_{'full' if full_finetuning else 'lora'}_{ckpt.split('/')[1]}_v1_{now}"

# Load environment variables
dotenv.load_dotenv()

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=False,
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    ckpt,
    quantization_config=bnb_config if not full_finetuning else None,
    device_map="auto",
    attn_implementation="eager",
)

tokenizer = AutoTokenizer.from_pretrained(ckpt)
tokenizer.pad_token = tokenizer.eos_token

# PEFT configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    task_type="CAUSAL_LM",
)

if not full_finetuning:
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

# Apply Gemma chat template
gemma_chat_template = {
    "chat_template": "{% for message in messages %}"
    "{{'<start_of_turn>' + message['role'] + '\n' + message['content'] + '<end_of_turn>\n'}}"
    "{% endfor %}"
}
tokenizer.chat_template = gemma_chat_template

# Load datasets
train = load_from_disk("../data/train")
test = load_from_disk("../data/validation")

with open("../prompt/prompt_v4.md", "r") as f:
    prompt = f.read()
    
def format_chat(batch):
    formatted = []
    for inp, out in zip(batch["input"], batch["output"]):
        messages = [
            {"role": "model", "content": prompt + f"----- Sentença judicial: {inp}"},
            {"role": "agent", "content": out}
        ]
        template = Template(tokenizer.chat_template["chat_template"])
        formatted.append(template.render(messages=messages))
    return {"text": formatted}
    return {"text": formatted}

train = train.map(format_chat, batched=True)
test = test.map(format_chat, batched=True)

train.drop_columns(["input", "output"])
# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_steps=5,
    num_train_epochs=num_train_epochs,
    learning_rate=2e-5,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir=f"outputs_{experiment_name}",
    report_to="wandb",
    save_strategy="steps",
    save_steps=10,
    save_total_limit=3,
    eval_strategy="steps",
    eval_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Response-only training mask
class ResponseMaskingCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        batch = super().__call__(features)
        response_start = "<start_of_turn>model\n"
        mask = []
        for text in [f["text"] for f in features]:
            response_pos = text.find(response_start)
            mask.extend([int(i >= response_pos) for i in range(len(text))])
        batch["labels"] = torch.where(
            torch.tensor(mask, dtype=torch.bool),
            batch["labels"],
            -100
        )
        return batch

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train,
    eval_dataset=test,
    # dataset_text_field="text",
    # max_seq_length=max_seq_length,
    data_collator=ResponseMaskingCollator(tokenizer),
    args=training_args,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)

# Initialize WandB
wandb.init(name=experiment_name)

# Training
print("Starting training...")
train_result = trainer.train()
wandb.finish()

# Save model
model.save_pretrained(f"{experiment_name}_model")
tokenizer.save_pretrained(f"{experiment_name}_tokenizer")

print("Training completed successfully!")