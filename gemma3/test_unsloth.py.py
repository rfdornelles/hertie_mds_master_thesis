from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_data_formats
from unsloth.chat_templates import train_on_responses_only
from unsloth.chat_templates import get_chat_template

from trl import SFTTrainer, SFTConfig 
from datasets import load_from_disk
import torch


ckpt = "unsloth/gemma-3-12b-it" #"unsloth/gemma-3-27b-it-GGUF" #"unsloth/gemma-3-27b-it-unsloth-bnb-4bit" #"google/gemma-3-4b-it"

model, tokenizer = FastModel.from_pretrained(
    model_name = ckpt,
    attn_implementation='eager',
    max_seq_length = 2048*8, # Choose any for long context!
    load_in_4bit = False,  # 4 bit quantization to reduce memory
    load_in_8bit = True, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = True, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
    device_map="auto",
)

torch.cuda.empty_cache()

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

train = load_from_disk("../data/train")
train = standardize_data_formats(train)

with open("prompt_v2.txt", "r") as f:
    prompt = f.read()

def build_template(batch):
    outputs = []
    # Itera sobre cada par input/output do batch
    for inp, out in zip(batch["input"], batch["output"]):
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': f"----- Sentença judicial:     {inp}"},
            {'role': 'agent', 'content': out}
        ]
        # Aplica o template para cada exemplo e adiciona o resultado na lista
        outputs.append(tokenizer.apply_chat_template(messages))
    return {"text": outputs}

# split train into train and validation
train_test = train.train_test_split(test_size=0.3, seed=42)

train = train_test["train"]
test = train_test["test"]

train = train.map(build_template, batched=True)
test = test.map(build_template, batched=True)


trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train,
    eval_dataset = test, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 30,
        learning_rate = 2e-6, # Reduce to 2e-5 for long training runs
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("gemma-3-12")  # Local saving
tokenizer.save_pretrained("gemma-3-12")
model.save_pretrained_merged("gemma-3-finetune-12b", tokenizer)

# test the model
test = load_from_disk("../data/test")
test = standardize_data_formats(test)
test = test.map(build_template, batched=True)

# results = trainer.predict(test)
# print(results)

input = test[0]['input']
tokenized_input = tokenizer(input, return_tensors="pt")
output = model.generate(**tokenized_input)
decoded_output = tokenizer.decode(output, skip_special_tokens=True)

tokenizer.batch_decode(output, skip_special_tokens=True)