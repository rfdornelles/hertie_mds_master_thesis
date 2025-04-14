import pandas as pd
import json
from collections import OrderedDict
from datasets import Dataset

# Lista com a ordem desejada das chaves
ordered_cols = [
    "processo",
    "juiz",
    "sexo_juiz",
    "nome",
    "local",
    "maconha",
    "maconha_g",
    "maconha_outras",
    "cocaina",
    "cocaina_g",
    "crack",
    "crack_g",
    "ecstasy",
    "ecstasy_g",
    "lsd",
    "lsd_g",
    "sentenca",
    "pena_base",
    "tot_pen",
    "tot_pen_meses",
    "outras_drogas",
    "resultado_art_28",
    "resultado_art_33",
    "resultado_art_34",
    "resultado_art_35",
    "denuncia_art_33",
    "denuncia_art_34",
    "denuncia_art_35",
    "flag_local_de_trafico",
    "flag_preso_no_momento_da_sentenca",
    "flag_confissao_informal",
    "flag_confissao",
    "flag_denuncia_anonima",
    "flag_denuncia",
    "flag_atitude_suspeita",
    "flag_divergencias_nos_relatos_dos_policiais",
    "flag_investigacao",
    "flag_interceptacao",
    "flag_mandado",
    "aval_antecedentes",
    "aval_conduta",
    "aval_personalidade",
    "aval_natureza",
    "aval_quantidade",
    "aval_variedade",
    "aval_circunstancias",
    "aval_consequencias",
    "aval_culpabilidade"
]

def row_to_ordered_json(row, ordered_cols):
    """Gera uma string JSON a partir de uma linha, preservando a ordem das chaves."""
    ordered_dict = OrderedDict()
    for col in ordered_cols:
        ordered_dict[col] = row[col]
    # Explicitamente não ordenar as chaves
    return json.dumps(ordered_dict, ensure_ascii=False, sort_keys=False)

def save_dataset(path, input_col='julgado', split=None):
    name = path.split('.parquet')[0]
    print(f"Processing {name} dataset...")

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f'Error: {e}')
        return

    # Converte as colunas em JSON manualmente, linha a linha, para garantir a ordem
    output_data = df.apply(lambda row: row_to_ordered_json(row, ordered_cols), axis=1)
    
    # Monta o DataFrame final com a coluna 'output' (única string) e as outras colunas que precisar
    final_df = pd.DataFrame({
        'processo': df['processo'],
        'input': df[input_col],
        'output': output_data
    })

    # Cria o Dataset usando from_pandas
    dataset = Dataset.from_pandas(final_df, split=split)

    try:
        dataset.save_to_disk(name)
        print(f'Successfully saved dataset {name}!')
    except Exception as e:
        print(f'Error: {e}')

# Testando com os arquivos
save_dataset('validation.parquet')
save_dataset('train.parquet')
save_dataset('test.parquet')
save_dataset('lacerda_clean.parquet')
