from transformers import pipeline
import pandas as pd
from tqdm import tqdm

## open llama model and ask to summarise a sentence
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

summarizer = pipeline("text-generation", model=MODEL_NAME)

## open dataset
df = pd.read_parquet("validation.parquet")

df = df[["id", "processo", "julgado"]]

prompt1 = "Resuma a sentença judicial a ser enviada a seguir informando seus principais aspectos como nome das partes, número do processo, as acusasões, os artigos de lei e o resultado."

def gen_summary(text, prompt = prompt1, max_new_tokens = 2048/2, temperature = 0.1, return_full_text = False, do_sample = False):
  
    # print(f"Summarising text: {text}")
    # print(f"Prompt: {prompt}")
    # print(f"Max new tokens: {max_new_tokens}")
    # print(f"Temperature: {temperature}")
    
    messages = [
     {"role": "system", "content": prompt},
     {"role": "user", "content": text}
    ]
  
    response = summarizer(messages, max_new_tokens = max_new_tokens, temperature = temperature, return_full_text = return_full_text, do_sample = do_sample)
    
    result = response[0]['generated_text']
    # print(result)
    
    return result

tqdm.pandas()


## test with a simple case
df["summarised"] = df["julgado"].progress_apply(gen_summary)

## description of the summarised text
# df['summ_lenght'] = df['summarised'].apply(len)
# df['original_lenght'] = df['julgado'].apply(len)
# df['delta'] = df['original_lenght'] - df['summ_lenght']
# df['delta'].describe()
# df['summ_lenght'].describe()
# df['original_lenght'].describe()

# save the experiment
df.to_parquet(f"validation_summarised_{MODEL_NAME.split('/')[1]}.parquet")

# df.sort_values("delta", ascending = False).head(10)

### trying a more complex case

# 

prompt2 = """Você é um assistente jurídico detalhista e precisa resumir sentenças judiciais complexas de tráfico de drogas. Para isso, você deve seguir um padrão de resumo que inclui informações específicas sobre o réu, os fatos do crime, as drogas apreendidas, a denúncia e os artigos legais, a sentença e a pena, flags relevantes e avaliações específicas.
---

## Instruções:

### 1. Identificação Geral (sempre incluir):
- **Número do Processo**
- **Nome e gênero do Juiz**
- **Vara criminal**

### 2. Identificação do Réu (para cada réu mencionado):
- Nome completo
- Informações sobre menoridade, adolescência ou nacionalidade (se relevantes)

### 3. Fatos do Crime (resumir claramente):
- **Local** (endereço ou bairro)
- **Data e horário**
- **Circunstâncias da abordagem policial** (motivação, atitude suspeita, fuga, etc.)

### 4. Drogas e Objetos Apreendidos (sempre listar claramente):
- Tipos e quantidades das drogas apreendidas (em gramas ou unidades)
- Objetos relacionados ao tráfico (balança, embalagens, dinheiro, etc.)

### 5. Denúncia e Artigos Legais (sempre resumir claramente):
- Trechos sobre a denúncia formal ou anônima
- Artigos específicos mencionados da Lei nº 11.343/06 ou outros relevantes

### 6. Sentença e Pena (sempre detalhar claramente):
- Pena total aplicada (tempo em anos/meses)
- Regime inicial (Fechado, Semi-aberto ou Aberto)
- Aplicação ou não do art. 33, §4º (redução de pena, réu primário, etc.)
- Substituição de pena (se houver)

### 7. Flags (sempre citar trechos relevantes se presentes):
- Confissão (formal/informal)
- Denúncia (anônima ou não)
- Investigação, interceptação, mandado
- Local associado ao tráfico
- Réu preso na sentença
- Revista vexatória
- Nacionalidade relevante
- Divergências nos depoimentos policiais

### 8. Avaliações Específicas (sempre registrar explicitamente se presentes):
- Antecedentes criminais
- Avaliação da conduta ou personalidade do réu
- Natureza do delito
- Quantidade ou variedade das drogas
- Circunstâncias ou consequências do crime
- Culpabilidade do réu
"""

df["summarised_2"] = df["julgado"].progress_apply(gen_summary, max_new_tokens = 2048, prompt = prompt2)


# save the experiment
df.to_parquet(f"validation_summarised_2_{MODEL_NAME.split('/')[1]}.parquet")

# ## check the experiment result
# df_results = pd.read_parquet("validation_summarised_2_{model_name}.parquet")

# df_results[["id", "processo", "julgado", "summarised", "summarised_2"]].head(10)