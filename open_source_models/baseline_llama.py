## test llama 3.2 on the dataset before ft
## make sure it yelds json output

### references:
# https://medium.com/@alejandro7899871776/structure-output-with-llama-from-scratch-39c487b6be81
# https://www.datacamp.com/tutorial/fine-tuning-llama-3-2
# https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/

# Load model directly
from transformers import pipeline
from datasets import load_from_disk
import torch
import json

torch.cuda.empty_cache()

# load datasets
validation = load_from_disk('../data/validation')

# load prompt
with open('../prompt/prompt_v4.md', 'r') as f:
    prompt = f.read()

# test
validation[1]['input']

model = 'meta-llama/Llama-3.2-3B-Instruct'

pipe = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "system", "content": prompt},
    {"role": "user", "content": validation[1]['input']},
]

outputs = pipe(
    messages,
    max_new_tokens=1000,
    # temperature=0.0,
    do_sample=False
)

output = outputs[0]["generated_text"][-1]['content']

output


# trying to design the object
from pydantic import BaseModel, Field
from typing import List, Optional
import unicodedata

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


## apply to the output

# SentencasJudiciaisResponse.model_validate_json(output)

from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object=SentencasJudiciaisResponse)
output_parsed = parser.parse(output)

pd.DataFrame(output_parsed)
