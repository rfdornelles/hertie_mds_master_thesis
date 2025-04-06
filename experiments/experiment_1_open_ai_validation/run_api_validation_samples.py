## objective: run trough the OpenAI api the validation samples
# using the chepeast model available (4o-mini)

import dotenv
import pandas as pd
import os
import re
from openai import OpenAI
import json
from datasets import load_from_disk

from pydantic import BaseModel, Field
from typing import List, Optional
import unicodedata



#### load environment variables from .env file
dotenv.load_dotenv()

API_KEY = os.getenv("HERTIE_OPENAI_API_KEY")

#### definitions
model = 'gpt-4o-mini'
experiment = 'experiment_1_open_ai_validation'

#### download data
df = load_from_disk("../../data/validation")
## get only the first 10 samples
df = df.select(range(10))

### load prompt
with open(f"../../prompt/prompt_v4_clean.md", "r") as f:
    prompt = f.read()
    
### define structure of the output

class SentencaJudicial(BaseModel):
    processo: str = Field(..., description='Número do processo (ex.: "00316684320178260050").')
    juiz: Optional[str] = Field(..., description="Nome do juiz responsável.")
    sexo_juiz: Optional[str] = Field(..., description='Gênero do juiz (ex.: "Masculino" ou "Feminino").')
    nome: str = Field(..., description="Nome completo do réu.")
    local: Optional[str] = Field(
        ...,
        description="Local relacionado ao crime/ocorrência, podendo ser: Comércio e serviços, Estádio, Favela - Viela, Invasão/Ocupação, Residência, Restaurante e afins, Terminal/Estação, Via Pública ou Área não ocupada."
    )
    
    maconha: bool = Field(..., description="True se houver apreensão de maconha; False caso contrário.")
    maconha_g: float = Field(..., description="Quantidade de maconha (gramas). 0 se não informado.")
    maconha_outras: bool = Field(..., description="True se há menção a outras formas de maconha (skank, haxixe); False caso contrário.")
    
    cocaina: bool = Field(..., description="True se houver apreensão de cocaína; False caso contrário.")
    cocaina_g: float = Field(..., description="Quantidade de cocaína em gramas. 0 se não informado.")
    
    crack: bool = Field(..., description="True se houver apreensão de crack; False caso contrário.")
    crack_g: float = Field(..., description="Quantidade de crack em gramas. 0 se não informado.")
    
    ecstasy: bool = Field(..., description="True se houver apreensão de ecstasy; False caso contrário.")
    ecstasy_g: float = Field(..., description="Quantidade de ecstasy em gramas. 0 se não informado.")
    
    lsd: bool = Field(..., description="True se houver apreensão de LSD; False caso contrário.")
    lsd_g: float = Field(..., description="Quantidade de LSD em gramas. 0 se não informado.")
    
    sentenca: Optional[str] = Field(..., description="Resultado final da sentença, podendo ser: Absolvição, Desclassificação, Parcialmente Procedente ou Procedente.")
    pena_base: Optional[str] = Field(..., description='Texto da pena-base (ex.: "8a" para 8 anos; "9a 6m" para 9 anos e 6 meses).')
    tot_pen: Optional[str] = Field(..., description='Pena total em texto (ex.: "10a 2m"), no formato <ANOS>a <MESES>m <DIAS>d.')
    tot_pen_meses: float = Field(..., description="Pena total convertida em meses. 0 se não informado.")
    
    outras_drogas: bool = Field(..., description="True se houver menção a outras substâncias além de maconha, cocaína, crack, ecstasy e LSD; False caso contrário.")
    
    resultado_art_28: bool = Field(..., description="True se o resultado/sentença fizer aplicação do art. 28 (uso pessoal); False se não.")
    resultado_art_33: bool = Field(..., description="True se o resultado/sentença fizer aplicação do art. 33 (tráfico); False se não.")
    resultado_art_34: bool = Field(..., description="True se o resultado/sentença fizer aplicação do art. 34; False se não.")
    resultado_art_35: bool = Field(..., description="True se o resultado/sentença fizer aplicação do art. 35 (associação); False se não.")
    
    denuncia_art_33: bool = Field(..., description="True se a denúncia alegar o art. 33; False se não.")
    denuncia_art_34: bool = Field(..., description="True se a denúncia alegar o art. 34; False se não.")
    denuncia_art_35: bool = Field(..., description="True se a denúncia alegar o art. 35; False se não.")
    
    flag_local_de_trafico: bool = Field(..., description="True se houver indicação de local associado a tráfico (ponto de venda, depósito, etc); False se não.")
    flag_preso_no_momento_da_sentenca: bool = Field(..., description="True se há indicação de que o réu estava ou permaneceu preso na ocasião da sentença.")
    flag_confissao_informal: bool = Field(..., description="True se houver menção de confissão de modo informal; False caso contrário.")
    flag_confissao: bool = Field(..., description="Desconsiderando confissões informais, True se houver menção a confissão formal; False se não.")
    flag_denuncia_anonima: bool = Field(..., description="True se houver menção a denúncia anônima; False caso contrário.")
    flag_denuncia: bool = Field(..., description="Após remover menções anônimas, True se houver denúncia formal; False caso contrário.")
    flag_atitude_suspeita: bool = Field(..., description="True se houver menção a atitude ou comportamento suspeito do réu; False se não.")
    flag_divergencias_nos_relatos_dos_policiais: bool = Field(..., description="True se houver divergências nos relatos dos policiais; False se não.")
    flag_investigacao: bool = Field(..., description="True se o texto mencionar investigação, inquérito ou diligências; False se não.")
    flag_interceptacao: bool = Field(..., description="True se houver menção a interceptações (telefônicas ou monitoramento); False se não.")
    flag_mandado: bool = Field(..., description="True se houver menção à expedição ou cumprimento de mandado; False se não.")
    
    aval_antecedentes: bool = Field(..., description="True se houver avaliação de antecedentes criminais; False se não.")
    aval_conduta: bool = Field(..., description="True se houver avaliação da conduta do réu; False se não.")
    aval_personalidade: bool = Field(..., description="True se o texto avaliar a personalidade do réu; False se não.")
    aval_natureza: bool = Field(..., description="True se houver avaliação da natureza do delito; False se não.")
    aval_quantidade: bool = Field(..., description="True se houver avaliação da quantidade de drogas ou itens apreendidos; False se não.")
    aval_variedade: bool = Field(..., description="True se houver avaliação da variedade de substâncias; False se não.")
    aval_circunstancias: bool = Field(..., description="True se o texto avaliar as circunstâncias do crime (local, condições etc.); False se não.")
    aval_consequencias: bool = Field(..., description="True se o texto abordar as consequências do delito (legais, sociais, pessoais); False se não.")
    aval_culpabilidade: bool = Field(..., description="True se houver menção da culpabilidade ou grau de reprovação; False se não.")

    # class Config:
    #     json_schema_extra = {
    #         "example": {
    #             "processo": "00316684320178260050",
    #             "juiz": "JOÃO DA SILVA",
    #             "sexo_juiz": "Masculino",
    #             "nome": "FULANO DE TAL",
    #             "local": "Via Pública",
    #             "maconha": True,
    #             "maconha_g": 150.0,
    #             "maconha_outras": True,
    #             "cocaina": False,
    #             "cocaina_g": 0,
    #             "crack": False,
    #             "crack_g": 0,
    #             "ecstasy": False,
    #             "ecstasy_g": 0,
    #             "lsd": False,
    #             "lsd_g": 0,
    #             "sentenca": "Procedente",
    #             "pena_base": "8a",
    #             "tot_pen": "8a 6m",
    #             "tot_pen_meses": 102,
    #             "outras_drogas": False,
    #             "resultado_art_28": False,
    #             "resultado_art_33": True,
    #             "resultado_art_34": False,
    #             "resultado_art_35": False,
    #             "denuncia_art_33": True,
    #             "denuncia_art_34": False,
    #             "denuncia_art_35": False,
    #             "flag_local_de_trafico": True,
    #             "flag_preso_no_momento_da_sentenca": True,
    #             "flag_confissao_informal": False,
    #             "flag_confissao": True,
    #             "flag_denuncia_anonima": False,
    #             "flag_denuncia": True,
    #             "flag_atitude_suspeita": False,
    #             "flag_divergencias_nos_relatos_dos_policiais": False,
    #             "flag_investigacao": True,
    #             "flag_interceptacao": False,
    #             "flag_mandado": False,
    #             "aval_antecedentes": True,
    #             "aval_conduta": True,
    #             "aval_personalidade": False,
    #             "aval_natureza": True,
    #             "aval_quantidade": True,
    #             "aval_variedade": False,
    #             "aval_circunstancias": False,
    #             "aval_consequencias": False,
    #             "aval_culpabilidade": False
    #         }
    #     }

class SentencasJudiciaisResponse(BaseModel):
    reus: List[SentencaJudicial] = Field(..., description="Lista de réus extraídos da sentença judicial.")

    # class Config:
    #     json_schema_extra = {
    #         "example": {
    #             "reus": [
    #                 {
    #                     "processo": "00316684320178260050",
    #                     "juiz": "JOÃO DA SILVA",
    #                     "sexo_juiz": "Masculino",
    #                     "nome": "FULANO DE TAL",
    #                     "local": "Via Pública",
    #                     "maconha": True,
    #                     "maconha_g": 150.0,
    #                     "maconha_outras": True,
    #                     "cocaina": False,
    #                     "cocaina_g": 0,
    #                     "crack": False,
    #                     "crack_g": 0,
    #                     "ecstasy": False,
    #                     "ecstasy_g": 0,
    #                     "lsd": False,
    #                     "lsd_g": 0,
    #                     "sentenca": "Procedente",
    #                     "pena_base": "8a",
    #                     "tot_pen": "8a 6m",
    #                     "tot_pen_meses": 102,
    #                     "outras_drogas": False,
    #                     "resultado_art_28": False,
    #                     "resultado_art_33": True,
    #                     "resultado_art_34": False,
    #                     "resultado_art_35": False,
    #                     "denuncia_art_33": True,
    #                     "denuncia_art_34": False,
    #                     "denuncia_art_35": False,
    #                     "flag_local_de_trafico": True,
    #                     "flag_preso_no_momento_da_sentenca": True,
    #                     "flag_confissao_informal": False,
    #                     "flag_confissao": True,
    #                     "flag_denuncia_anonima": False,
    #                     "flag_denuncia": True,
    #                     "flag_atitude_suspeita": False,
    #                     "flag_divergencias_nos_relatos_dos_policiais": False,
    #                     "flag_investigacao": True,
    #                     "flag_interceptacao": False,
    #                     "flag_mandado": False,
    #                     "aval_antecedentes": True,
    #                     "aval_conduta": True,
    #                     "aval_personalidade": False,
    #                     "aval_natureza": True,
    #                     "aval_quantidade": True,
    #                     "aval_variedade": False,
    #                     "aval_circunstancias": False,
    #                     "aval_consequencias": False,
    #                     "aval_culpabilidade": False
    #                 }
    #             ]
    #         }
    #     }

# Agora, a classe SentencasJudiciaisResponse contém uma lista de réus, que pode representar múltiplos réus extraídos da sentença.

# teste de chamada da API

client = OpenAI(api_key=API_KEY)

completion = client.beta.chat.completions.parse(
  model = model,
  temperature=0,
  max_tokens=1000,
  messages = [
    { "role": "system", "content": prompt },
    { "role": "system", "content": f"O número do processo deve ser: {json.dumps(df[0]['processo'])}"},
    { "role": "user", "content": json.dumps(df[0]['input']) }
  ],
  response_format = SentencasJudiciaisResponse
)

print(completion)
# Extrai o resultado parseado
parsed_response = completion.choices[0].message.parsed

# Cria um dicionário para a primeira linha (assumindo que a lista 'reus' não esteja vazia)
row_data = parsed_response.reus[0].model_dump() if parsed_response.reus else {}

# Cria um DataFrame com uma única linha a partir do dicionário
df_row = pd.DataFrame([row_data])

print(df_row)
# Converte o dataset Hugging Face para pandas e seleciona as colunas desejadas
df_output = pd.DataFrame(df['output'])
df_processo = pd.DataFrame(df['processo'])
df_expected = pd.concat([df_processo, df_output], axis=1).rename(columns={0: 'processo'})

# checar as colunas
columns = set(df_row.columns) & set(df_expected.columns)

# comparar

correct_fields = 0

for column in columns:
  
  def normalize_string(s: str) -> str:
    # Remove acentos e caracteres especiais, inclusive traços, espaços, etc.
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ascii', 'ignore').decode('utf-8')
    s = re.sub(r'[\W_]+', '', s)
    return s.upper()

  target_value = normalize_string(df_row[column].astype(str).values[0])
  expected_value = normalize_string(df_expected[column].astype(str).values[0])
  
  if target_value == expected_value:
    correct_fields += 1
    #print(f"Correct field: {column}")
  else:
    print(f"Incorrect field: {column}")
    print(f"Expected: {expected_value}")
    print(f"Got: {target_value}")
print(f"Correct fields: {correct_fields}/{len(columns)}")
