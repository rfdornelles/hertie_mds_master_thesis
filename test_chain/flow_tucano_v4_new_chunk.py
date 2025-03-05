import logging
import re
import tqdm
import pandas as pd
import numpy as np
import torch
import faiss

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from sentence_transformers import SentenceTransformer, util

import nltk

torch.cuda.empty_cache()

########################################
# Configuração de logging
########################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

########################################
# 1. Função de Chunking (segmentação)
########################################
# def chunk_text(text, chunk_size=2000, overlap=200):
#     """
#     Divide o texto em chunks de tamanho 'chunk_size' com sobreposição de 'overlap' caracteres.
#     """
#     text = re.sub(r'\s+', ' ', text).strip()
#     chunks = []
#     start = 0
#     n = len(text)
#     while start < n:
#         end = start + chunk_size
#         chunk = text[start:end]
#         chunks.append(chunk)
#         start = end - overlap
#         if start < 0:
#             start = 0
    
#     return chunks

def chunk_text(text, chunk_size=2000, overlap=200, similarity_threshold=0.75):
    """
    Divide o texto em chunks semânticos, mantendo o contexto e relevância.
    - max_chunk_size: Tamanho máximo do chunk em caracteres.
    - overlap: Sobreposição de caracteres entre chunks.
    - similarity_threshold: Limite de similaridade para fusão de sentenças em um chunk.
    """
    print("Iniciando chunking semântico...")
    
    # Limpeza básica do texto e divisão em sentenças
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # Se o chunk atual + próxima sentença ainda estiver no limite
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            # Adiciona o chunk atual à lista de chunks
            chunks.append(current_chunk.strip())
            
            # Inicia um novo chunk com sobreposição
            current_chunk = sentence

        # Verificação de similaridade para fusão semântica
        if len(chunks) > 1:
            prev_chunk = chunks[-1]
            embeddings = embedder.encode([prev_chunk, current_chunk])
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            
            if similarity > similarity_threshold:
                merged_chunk = f"{prev_chunk} {current_chunk}"
                chunks[-1] = merged_chunk
                current_chunk = ""
    
    # Adiciona o último chunk, se restar conteúdo
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print("Finalizado chunking semântico.")
    return chunks

########################################
# 2. Inicialização do modelo de embeddings
########################################
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
logger.info(f"Carregando modelo de embeddings: {embedding_model_name}")
embedder = SentenceTransformer(embedding_model_name)

########################################
# 3. Construção do índice FAISS a partir dos chunks
########################################
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, embeddings

########################################
# 4. Função de recuperação (retrieval)
########################################
def retrieve_chunks(query, index, chunks, top_k=3):
    q_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_embedding, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved

########################################
# 5. Prompt final (para extração campo a campo)
########################################
def build_prompt(retrieved_chunks, query):
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Você é um assistente especialista em Direito. Com base no contexto abaixo, responda de forma objetiva.

CONTEXT:
{context}

Pergunta: {query}
Resposta:"""
    return prompt

########################################
# 6. Inicialização do modelo LLM
########################################
llm_model_name = "TucanoBR/Tucano-2b4"
logger.info(f"Carregando modelo LLM: {llm_model_name}")
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto")

def generate_answer(prompt, max_new_tokens=150):
    """
    Gera a resposta do modelo LLM a partir do prompt.
    """
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Geração determinística (greedy)
            temperature=0.1,   # Baixa temperatura para respostas focadas
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.2,
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"[DEBUG] Resposta bruta do modelo:\n{answer}")
    
    # Extrair apenas o JSON, se possível
    try:
        json_start = answer.find("{")
        json_end = answer.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_str = answer[json_start:json_end]
            logger.info(f"[DEBUG] JSON extraído:\n{json_str}")
            return json_str
        else:
            logger.warning(f"[WARNING] JSON não encontrado na resposta: {answer}")
            return "None"
    except Exception as e:
        logger.error(f"[ERROR] Falha ao extrair JSON: {e}")
        return "None"
    
    
########################################
# 7. Mapeamento de campos (campo a campo)
########################################
field_prompts = {
    "processo": "Qual é o número do processo? (deve ter 20 dígitos)",
    "juiz": "Qual é o nome do juiz responsável?",
    "sexo_juiz": "Qual é o sexo do juiz? (Masculino ou Feminino)",
    "vara": "Qual é a vara judicial?",
    "nome": "Qual é o nome do réu?",
    "local": "Qual é o local do fato?",
    "maconha": "Qual é a informação sobre maconha?",
    "maconha_g": "Qual é a quantidade (em gramas) de maconha?",
    "cocaina": "Qual é a informação sobre cocaína?",
    "cocaina_g": "Qual é a quantidade (em gramas) de cocaína?",
    "crack": "Qual é a informação sobre crack?",
    "crack_g": "Qual é a quantidade (em gramas) de crack?",
    "ecstasy": "Qual é a informação sobre ecstasy?",
    "ecstasy_g": "Qual é a quantidade (em gramas) de ecstasy?",
    "lsd": "Qual é a informação sobre LSD?",
    "lsd_g": "Qual é a quantidade (em gramas) de LSD?",
    "outras": "Quais são as outras drogas mencionadas?",
    "anabolizantes": "O réu usou anabolizantes? (Sim ou Não)",
    "anorexigenos": "O réu usou anorexígenos? (Sim ou Não)",
    "haxixe": "O réu usou haxixe? (Sim ou Não)",
    "skank": "O réu usou skank? (Sim ou Não)",
    "lanca_perfume": "O réu usou lança perfume? (Sim ou Não)",
    "tolueno": "O réu usou tolueno? (Sim ou Não)",
    "den_drog": "Qual é a denúncia referente às drogas?",
    "den_outros": "Qual é a denúncia referente a outros crimes? (ou 'None' se não houver)",
    "sentenca": "Resuma a sentença.",
    "res_drogas": "Qual foi a resolução referente às drogas?",
    "res_outros": "Qual foi a resolução referente a outros crimes? (ou 'None' se não houver)",
    "pena_base": "Qual é a pena base?",
    "agravantes33_agrup": "Houve agravantes? (Sim ou Não)",
    "confissao": "O réu confessou? (Sim ou Não)",
    "menoridade": "O réu é menor de idade? (Sim ou Não)",
    "atenuantes33_agrup": "Houve atenuantes? (Sim ou Não)",
    "adolescente": "O réu é adolescente? (Sim ou Não)",
    "arma_de_fogo": "O réu possuía arma de fogo? (Sim ou Não)",
    "interestadual": "Houve casos interestaduais? (Sim ou Não)",
    "concurso_formal": "Houve concurso formal? (Sim ou Não)",
    "estabelecimento": "Houve estabelecimento de tráfico? (Sim ou Não)",
    "aumento33_agrup": "Houve aumento? (Sim ou Não)",
    "paragrafo_4o_agrupado": "Qual o conteúdo do parágrafo 4º agrupado?",
    "pena33": "Qual a pena aplicada conforme agrupamento 33?",
    "pena33_meses": "Qual a pena em meses conforme agrupamento 33?",
    "pena_drogas": "Qual a pena aplicada para drogas?",
    "pena_outros": "Qual a pena aplicada para outros crimes? (ou 'None' se não houver)",
    "tot_pen": "Qual a pena total?",
    "tot_pen_meses": "Qual a pena total em meses?",
    "substituicao_da_pena": "Houve substituição da pena? (ou 'None' se não houver)",
    "regime_inicial": "Qual o regime inicial?",
    "flag_local_de_trafico": "O caso envolve local de tráfico? (True ou False)",
    "flag_preso_no_momento_da_sentenca": "O réu estava preso no momento da sentença? (True ou False)",
    "flag_confissao_informal": "Houve confissão informal? (True ou False)",
    "flag_confissao": "Houve confissão formal? (True ou False)",
    "flag_denuncia_anonima": "Foi realizada denúncia anônima? (True ou False)",
    "flag_denuncia": "Houve denúncia? (True ou False)",
    "flag_atitude_suspeita": "Houve atitude suspeita? (True ou False)",
    "flag_divergencias_nos_relatos_dos_policiais": "Houve divergências nos relatos dos policiais? (True ou False)",
    "flag_investigacao": "Houve investigação? (True ou False)",
    "flag_interceptacao": "Houve interceptação? (True ou False)",
    "flag_mandado": "Houve mandado? (True ou False)",
    "flag_nacionalidade": "Houve nacionalidade mencionada? (True ou False)",
    "flag_revista_vexatoria": "Houve revista vexatória? (True ou False)",
    "aval_antecedentes": "Avalie os antecedentes do réu (True ou False).",
    "aval_conduta": "Avalie a conduta do réu (True ou False).",
    "aval_personalidade": "Avalie a personalidade do réu (True ou False).",
    "aval_natureza": "Avalie a natureza do delito (True ou False).",
    "aval_quantidade": "Avalie a quantidade (True ou False).",
    "aval_variedade": "Avalie a variedade (True ou False).",
    "aval_circunstancias": "Avalie as circunstâncias (True ou False).",
    "aval_consequencias": "Avalie as consequências (True ou False).",
    "aval_culpabilidade": "Avalie a culpabilidade (True ou False)."
}

########################################
# Nova função de similaridade
########################################
def select_relevant_chunks(chunks, query, top_k=3):
    # Mesma definição
    query_embedding = embedder.encode([query], convert_to_tensor=True)
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_indices = similarities.topk(k=top_k).indices
    return [chunks[i] for i in top_indices]

########################################
# 8. Função para processar o documento (campo a campo)
########################################
def process_document(doc_text, fields=field_prompts, top_k=3):
    chunks = chunk_text(doc_text)  # chunk semântico
    extracted = {}

    for field, query_prompt in fields.items():
        # Troca do retrieve_chunks para select_relevant_chunks
        relevant_chunks = select_relevant_chunks(chunks, query_prompt, top_k=top_k)
        prompt = build_prompt(relevant_chunks, query_prompt)
        answer = generate_answer(prompt)
        extracted[field] = answer

    return extracted

########################################
# Para extrair em grupos com JSON
########################################
def build_group_prompt(retrieved_chunks, group_name, field_list):
    """
    Cria um prompt que pede ao LLM para extrair todos os campos da lista 'field_list'
    com base nos 'retrieved_chunks' retornados pelo RAG.
    """
    context_text = "\n\n".join(retrieved_chunks)
    fields_str = ", ".join(field_list)

    prompt = f"""
<instruction>
Você é um assistente especialista em Direito.
A tarefa é extrair informações específicas do contexto fornecido e retornar apenas os dados em formato JSON.

**O que é um JSON:** 
JSON (JavaScript Object Notation) é um formato de dados baseado em pares chave-valor. Cada campo deve ter um valor correspondente, ou "None" caso o dado não esteja disponível.

**Exemplo de JSON válido:**
{{
    "campo1": "valor ou None",
    "campo2": "valor ou None",
    "campo3": "valor ou None"
}}

**Campos esperados para o grupo '{group_name}':**
{fields_str}

**Contexto para análise:**
{context_text}

**Instrução final:**
- Retorne apenas o JSON puro, sem explicações adicionais, códigos ou texto fora do JSON.
- Se não conseguir extrair algum campo, retorne "None" como valor.
- Não inclua tags <instruction> ou qualquer outro marcador.

Exemplo do formato esperado:
{{
    "maconha": "valor ou None",
    "maconha_g": "valor ou None",
    "cocaina": "valor ou None",
    "cocaina_g": "valor ou None",
    "crack": "valor ou None",
    "crack_g": "valor ou None",
    "ecstasy": "valor ou None",
    "ecstasy_g": "valor ou None",
    "lsd": "valor ou None",
    "lsd_g": "valor ou None",
    "outras": "valor ou None"
}}

Apenas o JSON, nada mais!
</instruction>
"""
    #logger.info(f"[DEBUG] Prompt enviado ao modelo:\n{prompt}")
    return prompt

def process_document_grouped(doc_text, grouped_fields_dict, top_k=3):
    chunks = chunk_text(doc_text, chunk_size=2000, overlap=200)
    index, _ = build_faiss_index(chunks)
    
    final_extracted = {}
    
    for group_name, fields_in_group in grouped_fields_dict.items():
        # Query “genérica” ou algo mais específico
        query_prompt = f"Informações específicas do grupo {group_name}."

        retrieved = retrieve_chunks(query_prompt, index, chunks, top_k=top_k)
        logger.info(f"[DEBUG] Grupo: {group_name} - retrieved chunks:\n{retrieved}\n")

        prompt = build_group_prompt(retrieved, group_name, fields_in_group)
        llm_response = generate_answer(prompt, max_new_tokens=800)  # menor max_new_tokens

        # Tentar extrair JSON
        logger.info(f"[DEBUG] Resposta bruta do LLM (grouped):\n{llm_response}\n")

        # 1) Tentar parse direto
        try:
            import json
            group_data = json.loads(llm_response)
            for field in fields_in_group:
                val = group_data.get(field, "None")
                final_extracted[field] = val
        except json.JSONDecodeError:
            logger.warning("[DEBUG] Falha ao fazer parse do JSON, tentaremos fallback com regex.")
            # 2) Tentar fallback para extrair algo que pareça JSON
            match = re.search(r"\{[\s\S]+\}", llm_response)
            if match:
                possible_json = match.group(0)
                try:
                    group_data = json.loads(possible_json)
                    for field in fields_in_group:
                        val = group_data.get(field, "None")
                        final_extracted[field] = val
                    continue
                except json.JSONDecodeError:
                    pass
            # Se chegou aqui, nada a fazer
            for field in fields_in_group:
                final_extracted[field] = "None"
    
    return final_extracted

########################################
# EXEMPLO DE USO
########################################
df = pd.read_parquet("../master_thesis/validation.parquet").head(5)
if "id" not in df.columns:
    df["id"] = df.index

grouped_fields = {
    "grupo_dados_processo": [
        "processo", "juiz", "sexo_juiz", "vara", "nome", "local"
    ],
    "grupo_drogas": [
        "maconha", "maconha_g", "cocaina", "cocaina_g", "crack",
        "crack_g", "ecstasy", "ecstasy_g", "lsd", "lsd_g", "outras"
    ],
    "grupo_outros_entorpecentes": [
        "anabolizantes", "anorexigenos", "haxixe", "skank",
        "lanca_perfume", "tolueno"
    ],
    "grupo_denuncia_sentenca": [
        "den_drog", "den_outros", "sentenca", "res_drogas", "res_outros"
    ],
    "grupo_pena_agravantes": [
        "pena_base", "agravantes33_agrup", "confissao", "menoridade",
        "atenuantes33_agrup", "adolescente", "arma_de_fogo",
        "interestadual", "concurso_formal", "estabelecimento",
        "aumento33_agrup", "paragrafo_4o_agrupado", "pena33",
        "pena33_meses", "pena_drogas", "pena_outros", "tot_pen",
        "tot_pen_meses", "substituicao_da_pena", "regime_inicial"
    ],
    "grupo_flags_avaliacoes": [
        "flag_local_de_trafico", "flag_preso_no_momento_da_sentenca",
        "flag_confissao_informal", "flag_confissao", "flag_denuncia_anonima",
        "flag_denuncia", "flag_atitude_suspeita", "flag_divergencias_nos_relatos_dos_policiais",
        "flag_investigacao", "flag_interceptacao", "flag_mandado", "flag_nacionalidade",
        "flag_revista_vexatoria", "aval_antecedentes", "aval_conduta",
        "aval_personalidade", "aval_natureza", "aval_quantidade", "aval_variedade",
        "aval_circunstancias", "aval_consequencias", "aval_culpabilidade"
    ]
}

results = []

for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Processando docs validação"):
    doc_text = row["julgado"]
    doc_id = row["id"]

    extracted_fields = process_document_grouped(doc_text, grouped_fields, top_k=3)

    out_dict = {"doc_id": doc_id}
    for group_name, fields_list in grouped_fields.items():
        for field in fields_list:
            gt_value = row.get(field, None)
            extracted_value = extracted_fields.get(field, "None")
            out_dict[f"{field}_gt"] = gt_value
            out_dict[f"{field}_extracted"] = extracted_value
    
    results.append(out_dict)

results_df = pd.DataFrame(results)
results_df.to_csv("/home/228446@hertie-school.lan/workspace/test_chain/resultados_extracao_validacao_grouped_tucano4.csv", index=False)
logger.info("Extração finalizada com sucesso!")

torch.cuda.empty_cache()





