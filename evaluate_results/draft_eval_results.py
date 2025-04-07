import json
import pandas as pd
from difflib import SequenceMatcher
from datasets import load_from_disk

# load data
df_expected = load_from_disk("../data/validation").to_pandas()

# extract data from json column
df_expected = df_expected.pop("output").apply(json.loads).apply(pd.Series)

#TODO: improve processo, since it's being used as id but it should be tested

# dados gerados
df_generated = pd.read_csv("../experiments/experiment_1_open_ai_validation/experiment_1_open_ai_validation-batch_results_parsed.csv")

# identify the process metadata
df_generated['processo'] = df_generated["custom_id"].apply(lambda x: x.split(":")[1])


# Identifica as colunas comuns (por nome) entre os dois DataFrames
columns_common = set(df_generated.columns) & set(df_expected.columns)

# Definição dos campos por tipo (ajuste conforme seu modelo)
numeric_fields = {"maconha_g", "cocaina_g", "crack_g", "ecstasy_g", "lsd_g", "tot_pen_meses"}
boolean_fields = {"maconha", "maconha_outras", "cocaina", "crack", "ecstasy", "lsd",
                  "outras_drogas", "resultado_art_28", "resultado_art_33", "resultado_art_34", "resultado_art_35",
                  "denuncia_art_33", "denuncia_art_34", "denuncia_art_35",
                  "flag_local_de_trafico", "flag_preso_no_momento_da_sentenca", "flag_confissao_informal",
                  "flag_confissao", "flag_denuncia_anonima", "flag_denuncia", "flag_atitude_suspeita",
                  "flag_divergencias_nos_relatos_dos_policiais", "flag_investigacao", "flag_interceptacao", "flag_mandado",
                  "aval_antecedentes", "aval_conduta", "aval_personalidade", "aval_natureza",
                  "aval_quantidade", "aval_variedade", "aval_circunstancias", "aval_consequencias", "aval_culpabilidade"}
text_fields = {"processo", "juiz", "sexo_juiz", "nome", "local", "sentenca", "pena_base", "tot_pen"}

# Função para normalizar texto: remove espaços, pontos e caracteres não alfanuméricos, e converte para maiúsculas
def normalize_text(text):
    return ''.join(char for char in str(text) if char.isalnum()).upper()

# Score para campos numéricos: se a diferença for zero, retorna 1, senão 0.
def score_numeric(gen, exp):
    try:
        diff = abs(float(gen) - float(exp))
    except Exception as e:
        return 0
    return 1 if diff == 0 else 0

# Score para campos booleanos: retorna 1 se iguais, 0 caso contrário.
def score_boolean(gen, exp):
    return 1 if str(gen).strip().upper() == str(exp).strip().upper() else 0

# Score para campos textuais: 
# Se os textos normalizados são iguais retorna 1.
# Se uma das strings estiver contida na outra ou a similaridade (SequenceMatcher) for acima de 0.8, retorna 0.5.
# Senão, retorna 0.
def score_text(gen, exp):
    norm_gen = normalize_text(gen)
    norm_exp = normalize_text(exp)
    if norm_gen == norm_exp:
        return 1
    elif norm_gen in norm_exp or norm_exp in norm_gen:
        return 0.5
    else:
        similarity = SequenceMatcher(None, norm_gen, norm_exp).ratio()
        return 1 if similarity >= 0.9 else (0.5 if similarity >= 0.8 else 0)

# Junta os DataFrames pelo campo "processo" para comparar os registros correspondentes
merged_df = pd.merge(df_generated, df_expected, on="processo", suffixes=("_gen", "_exp"))



# TODO: change processo as the common key, since it should be tested
#merged_df = pd.merge(df_generated, df_expected, left_on="processo_id", right_on= 'processo', suffixes=("_gen", "_exp"))

# Listas para armazenar scores por linha e por coluna
row_scores = []             # cada item: (processo, score, detalhes)
column_scores = {col: [] for col in columns_common}

# Itera sobre cada linha (processo) do DataFrame unido
for idx, row in merged_df.iterrows():
    processo = row["processo"]
    row_total = 0
    row_count = 0
    details = {}
    
    for col in columns_common:
        # Ignora o campo "processo", pois é usado apenas para o join
        if col == "processo":
            continue

        # Recupera o valor gerado e o esperado para o campo, usando os sufixos
        gen_val = row[f"{col}_gen"] if f"{col}_gen" in row else row[col]
        exp_val = row[f"{col}_exp"] if f"{col}_exp" in row else row[col]
        
        # Define a pontuação de acordo com o tipo do campo
        if col in numeric_fields:
            score = score_numeric(gen_val, exp_val)
        elif col in boolean_fields:
            score = score_boolean(gen_val, exp_val)
        elif col in text_fields:
            score = score_text(gen_val, exp_val)
        else:
            # Se não estiver categorizado, compara como texto
            score = score_text(gen_val, exp_val)
        
        details[col] = score
        row_total += score
        row_count += 1
        
        # Armazena a pontuação para essa coluna
        column_scores[col].append(score)
    
    # Nota por linha (processo): média dos scores da linha (entre 0 e 1)
    row_avg = row_total / row_count if row_count > 0 else 0
    row_scores.append({"processo": processo, "score": row_avg, "details": details})

# Nota geral: média de todas as pontuações
overall_score = sum(score for row in row_scores for score in row["details"].values()) / (len(row_scores) * row_count) if row_scores else 0

# Nota por coluna: média dos scores para cada coluna
column_avg_scores = {col: sum(scores)/len(scores) if scores else 0 for col, scores in column_scores.items()}

# Exibe os resultados
print("Pontuação por processo:")
for item in row_scores:
    print(f"Processo {item['processo']}: {item['score']*100:.2f}%")
    # Se desejar detalhes por coluna:
    # print(item["details"])
    
print("\nPontuação por coluna:")
for col, score in column_avg_scores.items():
    print(f"{col}: {score*100:.2f}%")
    
print(f"\nNota geral do modelo (F1 - % de exatidão): {overall_score*100:.2f}%")
