import os
import GPUtil

import logging
import time
from tqdm import tqdm
import re
import pandas as pd
import numpy as np
import torch
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


torch.cuda.empty_cache()


# Encontre o ID da GPU menos utilizada
def pick_least_used_gpu():
    # Obtenha a lista de GPUs
    devices = GPUtil.getGPUs()
    # Ordene pela menor memória usada
    device_min = min(devices, key=lambda x: x.memoryUsed)
    return device_min.id

# Defina a variável de ambiente CUDA_VISIBLE_DEVICES para usar apenas essa GPU
least_used_gpu_id = pick_least_used_gpu()
os.environ["CUDA_VISIBLE_DEVICES"] = str(least_used_gpu_id)
print(f"Usando GPU com ID {least_used_gpu_id} por estar menos utilizada.")

########################################
# Configuração de logging
########################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# OPCIONAL: CONFIGURAR INT8 OFFLOAD
# ----------------------------------------------------------------------------
# Se você está carregando o modelo em 8-bit e não couber inteiramente em GPU,
# pode ativar o offload parcial em CPU.
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,                 # Carregar em 8-bit
    llm_int8_enable_fp32_cpu_offload=True  # Habilita offload FP32 em CPU
)
 
llm_model_name = "ricdomolm/lawma-8b"
logger.info(f"Carregando modelo LLM: {llm_model_name}")

try:
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

    # device_map='auto' vai tentar alocar na GPU e, se faltar VRAM, offloadar o restante
    model = AutoModelForCausalLM.from_pretrained(
        llm_model_name,
        device_map="auto",
        # device_map="balanced_low_0",
        quantization_config=quant_config
    )
    logger.info("Modelo carregado com sucesso em 8-bit com offload habilitado.")
except ValueError as e:
    logger.error(f"Erro ao carregar modelo: {e}")
    raise e

# Confirma GPU
if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    logger.info(f"Executando na GPU {current_device}: {torch.cuda.get_device_name(current_device)}")
else:
    logger.warning("Sem GPU disponível, executando na CPU.")


########################################
# 1. Função de Chunking (segmentação)  #
########################################

def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Divide o texto em chunks de tamanho 'chunk_size' com sobreposição de 'overlap' caracteres.
    """
    print("Init chunking")
    text = re.sub(r'\s+', ' ', text).strip()
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
    
    print("End chunking")
    return chunks
  
  ####################################################
# 2. Inicialização do modelo de embeddings         #
####################################################
# Usaremos um modelo de SentenceTransformer que suporta múltiplos idiomas, incluindo o português.
embedding_model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
embedder = SentenceTransformer(embedding_model_name)
logger.info(f"Carregando modelo de embeddings: {embedding_model_name}")

####################################################
# 3. Construção do índice FAISS a partir dos chunks #
####################################################

def build_faiss_index(chunks):
    """
    Gera embeddings para cada chunk e constrói um índice FAISS.
    Retorna o índice e os embeddings gerados.
    """
    embeddings = embedder.encode(chunks, convert_to_numpy=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, embeddings
  
####################################################
# 4. Função de recuperação (retrieval) dos chunks   #
####################################################

def retrieve_chunks(query, index, chunks, top_k=3):
    """
    Dada uma query, gera seu embedding, busca os top_k chunks mais relevantes no índice
    e retorna a lista de chunks recuperados.
    """
    q_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_embedding, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    print(f"Retrieved chunks: {retrieved}")
    return retrieved

####################################################
# 5. Função para construir o prompt final          #
####################################################

def build_prompt(retrieved_chunks, query):
    """
    Monta um prompt que inclui as instruções, o contexto (chunks recuperados)
    e a query a ser respondida pelo LLM.
    """
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""Você é um assistente especialista em Direito. Com base no contexto abaixo, responda de forma objetiva.

CONTEXT:
{context}

Pergunta: {query}
Resposta:"""
    return prompt

####################################################
# 6. Inicialização do modelo LLM (TucanoBR/Tucano-2b4-Instruct)
####################################################

def generate_answer(prompt, max_new_tokens=4000):
    """
    Gera a resposta do modelo LLM a partir do prompt.
    """
    inputs = tokenizer(prompt, return_tensors='pt')
    # Envia para GPU (caso disponível)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        outputs = llm_model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.2)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Answer: {answer}")
    # Se houver marcador "Resposta:", extrai a parte após ele
    if "Resposta:" in answer:
        answer = answer.split("Resposta:")[-1].strip()
    return answer
  
  
####################################################
# 7. Mapeamento de campos para prompts específicos  #
####################################################
# Cada chave é um campo do seu modelo; o valor é a query que orienta o LLM para extrair aquela informação.
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

####################################################
# 8. Função para processar um documento            #
####################################################

def process_document(doc_text, fields=field_prompts, top_k=3):
    """
    Para um dado texto, realiza:
      - Chunking do texto
      - Construção do índice FAISS
      - Para cada campo, realiza recuperação dos chunks relevantes,
        monta o prompt e gera a resposta usando o LLM.
    Retorna um dicionário com as respostas extraídas para cada campo.
    """
    chunks = chunk_text(doc_text, chunk_size=2000, overlap=200)
    index, _ = build_faiss_index(chunks)
    
    extracted = {}
    for field, query_prompt in fields.items():
        retrieved = retrieve_chunks(query_prompt, index, chunks, top_k=top_k)
        prompt = build_prompt(retrieved, query_prompt)
        answer = generate_answer(prompt)
        extracted[field] = answer
    return extracted
  
  ####################################################
# 9. Carregamento do DataFrame com os dados         #
####################################################
# Supondo que o CSV possua colunas: "split", "texto" e os demais campos de ground truth.
df = pd.read_parquet("/home/228446@hertie-school.lan/workspace/master_thesis/validation.parquet")[0:5]

# Se houver coluna "id", ela será utilizada para identificação
if "id" not in df.columns:
    df["id"] = df.index
    

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quant_config = BitsAndBytesConfig(load_in_8bit=True)
llm_model = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    device_map="auto",
    quantization_config=quant_config
)

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

def build_group_prompt(retrieved_chunks, group_name, field_list):
    """
    Cria um prompt que pede ao LLM para extrair todos os campos da lista 'field_list'
    com base nos 'retrieved_chunks' retornados pelo RAG.
    """
    context_text = "\n\n".join(retrieved_chunks)
    fields_str = ", ".join(field_list)
    
    prompt = f"""Você é um assistente especialista em Direito.
Use o contexto abaixo para extrair os seguintes campos: {fields_str}.
Responda em JSON, usando as chaves exatamente como listadas e sem texto adicional.

CONTEXT:
{context_text}

REQUISITO:
Retorne um JSON puro com os campos do grupo {group_name} e seus valores.
Se algum campo não for encontrado, retorne "None" no lugar.
Exemplo de formato:
{{
    "campo1": "valor ou None",
    "campo2": "valor ou None",
    ...
}}

Por favor, só retorne o JSON.
"""
    return prompt

# def build_group_prompt(retrieved_chunks, group_name, field_list):
#     context_text = "\n\n".join(retrieved_chunks)
#     # Prepare a string de campos
#     fields_str = ", ".join(field_list)
#     # Construa um JSON de exemplo REAL
#     example_json_lines = []
#     for f in field_list:
#         example_json_lines.append(f'  "{f}": "None"')
#     example_json = "{\n" + ",\n".join(example_json_lines) + "\n}"

#     prompt = f"""Você é um assistente especialista em Direito.
# Use o contexto abaixo para extrair os seguintes campos: {fields_str}.

# Responda **exatamente** em JSON, usando as chaves listadas e sem texto adicional:
# {field_list}

# CONTEXT:
# {context_text}

# RETORNO ESPERADO:
# - Somente JSON, no seguinte formato:
# {example_json}

# - Se não encontrar o valor de algum campo, use \"None\" (string).

# Agora retorne somente o JSON, sem qualquer texto adicional.
# """
#     return prompt


def process_document_grouped(doc_text, grouped_fields_dict, top_k=3):
    """
    - Chunka o documento
    - Constrói índice
    - Para cada grupo de campos, faz retrieval e uma chamada ao LLM
      pedindo todos os campos em JSON
    Retorna um dicionário que mapeia cada campo => valor extraído.
    """
    # 1. Chunk + Index
    chunks = chunk_text(doc_text, chunk_size=2000, overlap=200)
    index, _ = build_faiss_index(chunks)
    
    final_extracted = {}
    
    # 2. Para cada grupo
    for group_name, fields_in_group in grouped_fields_dict.items():
        # Criamos uma "pergunta" que resume o que queremos para esses campos
        # Algo como "Extraia esses N campos: ..."
        # -> Precisamos construir a query (ou podemos usar algo genérico, ex.:
        #    "Extraia informações do grupo X"). A retrieval "genérica" pode
        #    não ser perfeita, mas vamos supor que a sentença seja toda relevante.
        
        # Query "genérica" ou focada
        query_prompt = f"Informações gerais do grupo {group_name}."
        
        # Recuperamos top_k chunks
        retrieved = retrieve_chunks(query_prompt, index, chunks, top_k=top_k)
        
        # Construímos um prompt pedindo *todos* os campos do grupo em JSON
        prompt = build_group_prompt(retrieved, group_name, fields_in_group)
        
        # Chamada ao LLM (apenas 1 vez para todo o grupo)
        llm_response = generate_answer(prompt)
        
        # 3. Tentar fazer parsing do JSON
        try:
            import json
            group_data = json.loads(llm_response)
            # group_data deve ser um dict com chaves = nomes dos campos
            # ex.: {"juiz": "Fulano", "sexo_juiz": "Masculino", ...}
            
            # 4. Incorporamos ao dicionário final
            for field in fields_in_group:
                val = group_data.get(field, "None")
                final_extracted[field] = val
        except json.JSONDecodeError:
            # se falhar o parse do JSON, podemos tentar fallback
            # ou marcar todos como None
            for field in fields_in_group:
                final_extracted[field] = "None"
    
    return final_extracted

# Exemplo dentro do loop de documentos
results = []

for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processando docs validação"):
    doc_text = row["julgado"]
    doc_id = row["id"]

    # Extraímos todos os campos em grupos
    extracted_fields = process_document_grouped(doc_text, grouped_fields, top_k=3)

    # Monta dict final com ground truth x extraído
    out_dict = {"doc_id": doc_id}
    for group_name, fields_list in grouped_fields.items():
        for field in fields_list:
            gt_value = row.get(field, None)
            extracted_value = extracted_fields.get(field, "None")
            out_dict[f"{field}_gt"] = gt_value
            out_dict[f"{field}_extracted"] = extracted_value
    
    results.append(out_dict)

results_df = pd.DataFrame(results)
results_df.to_csv("resultados_extracao_validacao_grouped_lawma.csv", index=False)
