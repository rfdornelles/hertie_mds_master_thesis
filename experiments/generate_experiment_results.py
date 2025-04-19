## objective: read the results of the experiments and generate the .parquet results

import os
import pandas as pd
from pydantic import BaseModel, Field
from typing import List, Optional
import unicodedata
from langchain_core.output_parsers import JsonOutputParser
import json
from json import JSONDecodeError
from tqdm import tqdm
import re

## definitions

experiments = [
  'experiment_gemma3_1b_it_baseline',
  'experiment_gemma3_12b_it_baseline_v2',
  'experiment_gemma3_27b_it_baseline_v2',
  'experiment_llama_3.1_8b_baseline',
  'experiment_llama_3.2_3b_baseline',
  'experiment_llama_3.2_3B_baseline_ft_unsloth_2025-04-13_17-52-07',
  'experiment_phi4_4b_baseline_unsloth',
  'experiment_phi4_4b_finetune_unsloth',
  'experiment_finetune_lora_master_thesis_tucanobr_2b4_ft_v1',
  'experiments/run_unsloth_ft_lora_master_thesis_lawma8b_ft_v1'
  ]

folder = "."



## 
print("Initializing the script... building the parser.")

## define the pydantic model for the output
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

class SentencasJudiciaisResponse(BaseModel):
    reus: List[SentencaJudicial] = Field(..., description="Lista de réus extraídos da sentença judicial.")

## init the parser

try:
  parser = JsonOutputParser(pydantic_object=SentencasJudiciaisResponse)
  print("Parser initialized successfully!")
  
except Exception as e:
  print(f"Error initializing parser: {e}")


# function to read each output
def parse_result(experiment, folder, file):

  processo = file.split(".")[0]
  full_file = f"{folder}/{experiment}/{file}"
  
  if (os.path.exists(full_file) == False):
    print(f"File {full_file} does not exist.")
    return None
  
  try:
    with open(full_file, "r", encoding="utf-8") as f:
      content = f.read()
    
  except Exception as e:
    print(f"Error reading file {full_file}: {e}")
    return None
    
  try:
    output = parser.parse(content)

  except Exception as e:
    
    # trying to clean the content before parsing again
    try:
      content = re.sub(r'^```(?:json)?\s*', '', content)
      content = re.sub(r'\s*```$', '', content)
      
      # normalize the booleans
      content = re.sub(r'(?i)\bTrue\b', 'true', content)
      content = re.sub(r'(?i)\bFalse\b', 'false', content)
      content = re.sub(r'(?i)\bNone\b', 'null', content)
      
      # try again
      output = parser.parse(content)
      
    except Exception as e:
        print(f"Error parsing file {full_file} after cleaning:")
        data = {field: None for field in SentencaJudicial.model_fields.keys()}
        data["processo"] = processo
    
        return pd.DataFrame([data])    
  
  
  try:
    # squish the results in a single row
    # Assuming 'output' is the list returned by your parser:
    merged_output = {}
    
    for item in output:
      merged_output.update(item)

    result = pd.DataFrame([merged_output])
    # force processo to be the name of the file
    result["processo"] = processo
  
  except Exception as e:    
    try:
      result = pd.DataFrame([output])
    except Exception as e:
      print(f"Error converting to DataFrame {full_file}: {e}")
      return None
  
  return result

## function to read each 
def read_experiment(experiment, overwrite=False):
  
  output_file = f"{folder}/{experiment}/{experiment}_results_parsed.parquet"
  # check if the file already exists
  if os.path.exists(output_file) and not overwrite:
    print(f"File {output_file} already exists. Skipping...")
    return pd.read_parquet(output_file)

  # read the files in the experiment folder
  try:
    files = os.listdir(f"{folder}/{experiment}")
    
  except Exception as e:
    print(f"Error reading directory {folder}/{experiment}: {e}")
    return None
  
  # only if the file ends with .json
  files = [f for f in files if f.endswith(".json")]

  results = []
  for file in tqdm(files):
    
    result = parse_result(experiment, folder, file)
    if result is not None:
      results.append(result)
  
  # concatenate all results into a single DataFrame
  if len(results) == 0:
    print(f"No results found for {experiment}")
    return None
  try:
    results = pd.concat(results, ignore_index=True)
  except Exception as e:
    print(f"Error concatenating results for {experiment}: {e}")
    return None
  
  
  # save the results to a parquet file
  try:
    results.to_parquet(output_file, index=False)
    
  except Exception as e:
    print(f"Error saving results to parquet for {experiment}: {e}")
    return None
  
  print(f"Experiment {experiment} successfully completed. Results saved to {output_file}")
  
## iterate over the experiments and read each one
for experiment in experiments:
  read_experiment(experiment, overwrite=True)
  
print("All experiments completed.")