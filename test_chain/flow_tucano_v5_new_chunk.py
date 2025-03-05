import logging
import re
import tqdm
import pandas as pd
import numpy as np
import torch
import faiss
import nltk
import json

from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer, util

# Se necessário, baixe os dados de tokenização do NLTK
nltk.download('punkt')

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
# 1. Função de Chunking Semântico
########################################
def chunk_text(text, chunk_size=4000, overlap=200, similarity_threshold=0.75):
    """
    Divide o texto em chunks semânticos, mantendo o contexto e relevância.
    - chunk_size: Tamanho máximo do chunk em caracteres.
    - overlap: Sobreposição de caracteres entre chunks.
    - similarity_threshold: Limite de similaridade para fusão de sentenças.
    """
    logger.info("Iniciando chunking semântico...")
    
    # Limpeza básica e divisão em sentenças
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
            # Inicia novo chunk com a sentença atual
            current_chunk = sentence

        # Se houver pelo menos um chunk, verifica similaridade com o último chunk
        if chunks:
            prev_chunk = chunks[-1]
            embeddings = embedder.encode([prev_chunk, current_chunk])
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            if similarity > similarity_threshold:
                merged_chunk = f"{prev_chunk} {current_chunk}"
                chunks[-1] = merged_chunk
                current_chunk = ""
    
    # Adiciona o último chunk, se houver conteúdo
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    logger.info("Finalizado chunking semântico.")
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
# 4. Função de recuperação (retrieval) de chunks relevantes
########################################
def retrieve_chunks(query, index, chunks, top_k=3):
    q_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(q_embedding, top_k)
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved

########################################
# 5. Montagem do prompt final para o LLM
########################################
def build_prompt(retrieved_chunks, field_name, query):
    """
    Monta o prompt para o LLM, instruindo-o a responder exclusivamente em JSON.
    O formato de resposta deve ser: {"<field_name>": "conteúdo extraído ou None"}
    """
    context = "\n\n".join(retrieved_chunks)
    
    # Exemplo baseado no próprio 'field_name', para reforçar que a chave do JSON deve ser o campo
    example_json = f'{{"{field_name}": "Exemplo de resposta"}}'
    
    prompt = f"""<instruction>Você é um assistente especialista em Direito. Com base no contexto abaixo, responda de forma objetiva.

Instruções:
- Você receberá um CONTEXTO sobre um caso jurídico.
- Com base no contexto, responda **exclusivamente** no seguinte formato JSON
- Exemplo de resposta para este campo específico:
{example_json}
- Perceba que a chave do JSON **deve ser** {field_name}.
- Não inclua outras chaves ou texto fora do JSON.

CONTEXTO:
{context}

</instruction>

'{{"{field_name}": "<SUA RESPOSTA AQUI>"}}' """
    return prompt


########################################
# 6. Inicialização do modelo LLM
########################################
llm_model_name = "TucanoBR/Tucano-2b4-Instruct"
logger.info(f"Carregando modelo LLM: {llm_model_name}")
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name, device_map="auto")

def generate_answer(prompt, max_new_tokens=200):
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
            # temperature=0.0001,
            top_k=50,
            top_p=1.0,
            repetition_penalty=1.2,
            renormalize_logits=True,
            # use_cache=True,
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"[DEBUG] Resposta bruta do modelo:\n{outputs}")
    
    # Tenta extrair um JSON se presente
    try:
        json_start = answer.find("{")
        json_end = answer.rfind("}") + 1
        if json_start != -1 and json_end != -1:
            json_str = answer[json_start:json_end]
            logger.info(f"[DEBUG] JSON extraído:\n{json_str}")
            return json_str
        else:
            return answer.strip()
    except Exception as e:
        logger.error(f"[ERROR] Falha ao extrair JSON: {e}")
        return answer.strip()

########################################
# 7. Função para processar o documento (pergunta campo a campo)
########################################
def process_document(doc_text, fields):
    """
    Para um documento (texto da sentença), realiza:
      1. Chunking semântico
      2. Construção do índice FAISS
      3. Para cada campo, realiza a recuperação e gera a resposta via LLM
    Retorna um dicionário com as respostas extraídas.
    """
    chunks = chunk_text(doc_text)
    index, _ = build_faiss_index(chunks)
    extracted = {}
    
    for field, query_prompt in fields.items():
        relevant_chunks = retrieve_chunks(query_prompt, index, chunks, top_k=3)
        prompt = build_prompt(relevant_chunks, field, query_prompt)
        answer = generate_answer(prompt)
        extracted[field] = answer
    return extracted

########################################
# 8. Definição dos campos para extração
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
# 9. Script principal para processar os documentos
########################################
def main():
    # Carregue seu dataset – exemplo usando um arquivo Parquet
    df = pd.read_parquet("../master_thesis/validation.parquet").head(5)
    if "id" not in df.columns:
        df["id"] = df.index

    results = []
    
    for idx, row in tqdm.tqdm(df.iterrows(), total=df.shape[0], desc="Processando documentos"):
        doc_text = row["julgado"]
        doc_id = row["id"]

        extracted_fields = process_document(doc_text, field_prompts)
        
        # Monta um dicionário com os resultados para o documento
        out_dict = {"doc_id": doc_id}
        for field, answer in extracted_fields.items():
            gt_value = row.get(field, None)
            out_dict[f"{field}_gt"] = gt_value
            out_dict[f"{field}_extracted"] = answer
        results.append(out_dict)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv("/home/228446@hertie-school.lan/workspace/test_chain/flow_tucano_v5_new-resultados_extracao_validacao.csv", index=False)
    logger.info("Extração finalizada com sucesso!")
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
