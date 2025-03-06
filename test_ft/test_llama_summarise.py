from transformers import pipeline
import pandas as pd
from tqdm import tqdm

## open llama model and ask to summarise a sentence
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"


summarizer = pipeline("text-generation", model=MODEL_NAME)

## open dataset
df = pd.read_parquet("validation.parquet")

## example
example = df["julgado"].iloc[0]

prompt1 = "Resuma a sentença judicial a ser enviada a seguir informando seus principais aspectos como nome das partes, número do processo, as acusasões, os artigos de lei e o resultado."

def gen_summary(text, prompt = prompt1, max_new_tokens = 2048/2, temperature = 0.1, return_full_text = False, do_sample = False):
  
    messages = [
     {"role": "system", "content": prompt},
     {"role": "user", "content": example}
    ]
  
    response = summarizer(text, max_new_tokens = max_new_tokens, temperature = temperature, return_full_text = return_full_text, do_sample = do_sample)
    return response[0]['generated_text']

tqdm.pandas()

df["summarised"] = df["julgado"].progress_apply(gen_summary)

## description of the summarised text
df['summ_lenght'] = df['summarised'].apply(len)
df['original_lenght'] = df['julgado'].apply(len)
df['delta'] = df['original_lenght'] - df['summ_lenght']
df['delta'].describe()
df['summ_lenght'].describe()
df['original_lenght'].describe()

# save the experiment
df.to_parquet("validation_summarised_{model_name}.parquet")

df.sort_values("delta", ascending = False).head(10)

prompt2 =  """Objetivo
Extraia, de sentenças judiciais criminais (Brasil) envolvendo tráfico de drogas ou correlatos, todas as informações relevantes para preencher posteriormente um conjunto de campos estruturados. Cada réu identificado no texto deve ter seu próprio bloco ou seção, preservando o máximo de detalhes.

Instruções
Dividir por Réu

Se existirem múltiplos réus no documento, crie uma seção separada para cada um, mantendo o número de processo se for o mesmo.
Manter Detalhes Essenciais

Registre quantidades, datas, drogas, artigos de lei e qualquer trecho que indique:
Apreensão de drogas (maconha, cocaína, crack, etc.)
Possível confissão (formal ou informal)
Denúncia (anônima ou formal)
Investigação, interceptações ou mandados
Flags de conduta (atitude suspeita, divergências em depoimentos)
Avaliações (antecedentes, culpabilidade, natureza do crime, etc.)
Se algo não constar, escreva “Não consta” ou “Não se aplica”.
Estrutura do Resumo
Para cada réu, organize as informações em tópicos, por exemplo:

Número do Processo: transcrever ou aproximar, se houver.
Juiz/Vara: nomes, menções sobre gênero do juiz, vara criminal.
Identificação do Réu: nome completo, menções à menoridade.
Local/Contexto: onde ocorreu, se for relevante (ponto de tráfico, etc.).
Drogas Apreendidas: tipos (maconha, cocaína, crack, ecstasy, LSD...) e quantidades (aproximadas em gramas). Anotar também se houve anabolizantes, haxixe, skank, lança-perfume, tolueno ou anorexígenos.
Denúncia/Artigos: trechos sobre denúncia por Lei de Drogas e/ou outros artigos.
Sentença/Pena: pena base, regime inicial, substituições, total em meses, etc.
Possíveis “Flags”: transcreva partes que apontem para:
Local de tráfico
Réu preso no momento da sentença
Confissão (formal ou informal)
Denúncia (anônima ou não)
Atitude suspeita
Divergências nos relatos policiais
Existência de investigação/interceptação
Mandado
Nacionalidade relevante
Revista vexatória
Possíveis “Avaliações”: menções a antecedentes, conduta, personalidade, quantidade/variedade de drogas, consequências e culpabilidade.
Se o Documento For Extenso

Mantenha os trechos importantes que mostram a relação entre o réu, o crime e as circunstâncias (tipo de droga, quantidade, depoimentos, etc.).
Resuma apenas trechos irrelevantes ou repetitivos, mas sem perder informações cruciais.
Forma de Entrega

Ao final, produza um texto estruturado com uma seção para cada réu, cobrindo todos os tópicos acima.
Marque claramente quando uma informação não aparece no texto.
Exemplo de Organização de Saída
markdown
Copy
Edit
[Resumo da Sentença]

**RÉU 1:**
1. Número do Processo: ...
2. Juiz/Vara: ...
3. Identificação do Réu: ...
4. Local/Contexto: ...
5. Drogas Apreendidas: ...
6. Denúncia/Artigos: ...
7. Sentença/Pena: ...
8. Flags: (local de tráfico, confissão, etc. + citações)
9. Avaliações: (antecedentes, conduta, etc. + citações)

**RÉU 2:** (mesma estrutura)

[Fim]
"""

df["summarised_2"] = df["julgado"].progress_apply(gen_summary, max_new_tokens = 2048, prompt = prompt2)

# save the experiment
df.to_parquet(f"validation_summarised_2_{MODEL_NAME}.parquet")
