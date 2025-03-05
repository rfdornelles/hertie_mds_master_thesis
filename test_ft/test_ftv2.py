# -*- coding: utf-8 -*-
"""
Script de Fine-Tuning para Modelo de Extração de Informações Jurídicas
Autor: Seu Nome
Data: [Data]
Descrição: Fine-tuning do modelo Llama para transformar sentenças judiciais em JSON estruturado
"""

import os
import json
import time
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

# Configurações Globais
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
CONTEXT_WINDOW = 4096  # Ajustar conforme o modelo
DATASET_PATH = 'validation.parquet'
OUTPUT_DIR = "./results"

# Helper Functions
def carregar_dados():
    """Carrega e prepara os dados do dataset"""
    df = pd.read_parquet(DATASET_PATH)
    
    # Verificar colunas necessárias
    colunas_necessarias = ['julgado', 'id'] + [c for c in df.columns if c not in ['julgado', 'id']]
    if not all(col in df.columns for col in colunas_necessarias):
        raise ValueError("Dataset não contém colunas necessárias")
    
    # Criar coluna com JSON de features
    df['data'] = df.drop(columns=['julgado', 'id']).apply(lambda x: x.to_json(), axis=1)
    return df[['julgado', 'data']]

def formatar_prompt(sentenca):
    """Formata o prompt com instruções claras"""
    return f"""<|begin_of_text|>
    TAREFA: Extrair informações estruturadas de sentenças judiciais.
    FORMATO DE SAÍDA: Apenas JSON válido.
    ---
    SENTENÇA: {sentenca}
    ---
    RESPOSTA: """

# Classe de Pré-processamento
class LegalJSONProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def preprocess(self, examples):
        """Processa batch de exemplos"""
        full_texts, respostas = [], []
        
        for sentenca, dados in zip(examples["julgado"], examples["data"]):
            prompt = formatar_prompt(sentenca)
            full_text = prompt + dados + "<|end_of_text|>"
            full_texts.append(full_text)
            respostas.append(dados)
        
        # Tokenização
        tokenized = self.tokenizer(
            full_texts,
            max_length=CONTEXT_WINDOW,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Criação de labels
        labels = tokenized["input_ids"].clone()
        for idx, text in enumerate(full_texts):
            # Calcular posição da resposta
            prompt_tokens = self.tokenizer.encode(formatar_prompt(""), add_special_tokens=False)
            answer_start = len(prompt_tokens)
            
            # Mascarar prompt e padding
            labels[idx, :answer_start] = -100
            labels[idx][tokenized["attention_mask"][idx] == 0] = -100
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
            "respostas": respostas
        }

# Classe de Treino Customizada
class LegalTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Calcula loss com pesos para elementos JSON"""
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        
        # Configurar pesos
        device = outputs.logits.device
        weights = torch.ones_like(inputs["labels"], dtype=torch.float32).to(device)
        
        # Tokens importantes do JSON
        tokens_especiais = {
            '{', '}', ':', ',', '[', ']', '"'
        }
        for token in tokens_especiais:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                weights[inputs["labels"] == token_id] = 2.0
        
        # Cálculo manual da loss
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()
        shift_weights = weights[..., 1:].contiguous()
        
        loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        per_token_loss = loss_fn(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
        # Aplicar pesos e normalizar
        weighted_loss = (per_token_loss * shift_weights.view(-1))
        total_loss = weighted_loss.sum()
        total_weights = shift_weights.sum()
        
        # Evitar divisão por zero
        final_loss = total_loss / total_weights if total_weights > 0 else total_loss
        
        return (final_loss, outputs) if return_outputs else final_loss

# Callbacks de Monitoramento
class MonitorLegal(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.log_file = os.path.join(OUTPUT_DIR, "training_logs.txt")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log detalhado em arquivo e console"""
        log_str = ""
        if 'loss' in logs:
            log_str += f"[Treino] Step {state.global_step} - Loss: {logs['loss']:.4f}\n"
        if 'eval_loss' in logs:
            log_str += f"[Validação] Step {state.global_step} - Loss: {logs['eval_loss']:.4f}\n"
        
        if log_str:
            print(log_str)
            with open(self.log_file, "a") as f:
                f.write(f"{time.ctime()} - {log_str}")
    
    def on_evaluate(self, args, state, control, **kwargs):
        """Gera exemplos durante a validação"""
        model = kwargs['model']
        eval_dataset = kwargs['eval_dataset']
        
        print("\nExemplos de Validação:")
        for i in range(3):
            sample = eval_dataset[i]
            inputs = {k: v.unsqueeze(0).to(model.device) for k, v in sample.items() if k != 'respostas'}
            
            generated = model.generate(
                inputs["input_ids"],
                max_length=CONTEXT_WINDOW,
                num_beams=3
            )
            
            output_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            try:
                json.loads(output_text.split("RESPOSTA:")[-1])
                status = "✅ JSON Válido"
            except:
                status = "❌ JSON Inválido"
            
            print(f"\nExemplo {i+1}:")
            print(f"Entrada: {sample['julgado'][:100]}...")
            print(f"Saída: {output_text[:200]}...")
            print(status)

# Função Principal
def main():
    # Configuração Inicial
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Carregar dados
    dados = carregar_dados()
    dataset = Dataset.from_pandas(dados).train_test_split(test_size=0.1)
    
    # Inicializar Modelo e Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['<|begin_of_text|>', '<|end_of_text|>']})
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=False  # Necessário para gradient checkpointing
    )
    model.resize_token_embeddings(len(tokenizer))
    
    # Pré-processamento
    processor = LegalJSONProcessor(tokenizer)
    tokenized_dataset = dataset.map(
        processor.preprocess,
        batched=True,
        batch_size=4,
        remove_columns=dataset["train"].column_names
    )
    
    # Configuração do Treino
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        report_to="none",
        optim="adamw_torch",
        max_grad_norm=1.0
    )
    
    # Inicializar Trainer
    trainer = LegalTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        callbacks=[MonitorLegal(tokenizer)]
    )
    
    # Treinar
    print("\nIniciando treinamento...")
    train_result = trainer.train()
    
    # Salvar resultados
    model.save_pretrained(os.path.join(OUTPUT_DIR, "modelo_final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "modelo_final"))
    
    print("\nTreinamento concluído!")
    print(f"Loss final: {train_result.metrics['train_loss']:.4f}")

if __name__ == "__main__":
    main()
    
def plot_progress():
    import matplotlib.pyplot as plt
    with open(os.path.join(OUTPUT_DIR, "training_logs.txt")) as f:
        lines = f.readlines()
    
    steps, train_loss, eval_loss = [], [], []
    for line in lines:
        if "Treino" in line:
            parts = line.split("Loss: ")
            steps.append(int(parts[0].split("Step ")[1].split(" -")[0]))
            train_loss.append(float(parts[1]))
        elif "Validação" in line:
            parts = line.split("Loss: ")
            eval_loss.append(float(parts[1]))
    
    plt.figure(figsize=(10,5))
    plt.plot(steps, train_loss, label='Train Loss')
    plt.plot(steps[:len(eval_loss)], eval_loss, label='Eval Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, "progresso.png"))
    plt.show()