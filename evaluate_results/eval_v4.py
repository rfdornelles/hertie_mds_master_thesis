## Objective: evaluate all experiments results

import os
import json
import pandas as pd
from datasets import load_from_disk
from difflib import SequenceMatcher
import numpy as np
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


### List all experiments and its results
experiments = []

print("Listing all experiments and its results...")

# iterate over each folder in the experiments folder
for root, dirs, files in os.walk("../experiments", topdown=True, onerror=None, followlinks=False):
  # iterate over each file in the folder
  for file in files:
    # check if the file is a parquet file
    if file.endswith("results_parsed.parquet"):
      # get the full path of the experiment file
      full_path = os.path.join(root, file)
      
      # ignore validation tests
      if full_path.__contains__("validation"):
        continue
      # add the experiment full path to the list
      experiments.append(full_path)
      # print(f"  Experiment: {full_path}")

print(f"  Found {len(experiments)} experiments.")
### load the validation data
print("Loading test data...")
df_test = load_from_disk("../data/test").to_pandas().pop("output").apply(json.loads).apply(pd.Series)

### determine the types of columns -- to each type we'll have a different evaluation
# numeric: they will be substracted
# boolean: exact match
# categorical: exact match
# open_textual: get the edit distance between them to determine the similarity

numeric_fields = ['maconha_g', 'cocaina_g', 'crack_g','ecstasy_g',
                  'lsd_g', 'tot_pen_meses']

boolean_fields = ['maconha', 'maconha_outras','cocaina', 'crack', 'ecstasy', 
                  'lsd',   'outras_drogas', 'resultado_art_28', 'resultado_art_33', 'resultado_art_34', 'resultado_art_35', 'denuncia_art_33', 'denuncia_art_34', 'denuncia_art_35', 'flag_local_de_trafico', 'flag_preso_no_momento_da_sentenca', 'flag_confissao_informal', 'flag_confissao', 'flag_denuncia_anonima', 'flag_denuncia','flag_atitude_suspeita','flag_divergencias_nos_relatos_dos_policiais','flag_investigacao', 'flag_interceptacao', 'flag_mandado', 'aval_antecedentes', 'aval_conduta', 'aval_personalidade', 'aval_natureza', 'aval_quantidade', 'aval_variedade', 'aval_circunstancias', 'aval_consequencias', 'aval_culpabilidade']
# sentence length is going to be categorical since it should be exact match
categorical_fields = ['sexo_juiz', 'local',  'sentenca', 'pena_base', 'tot_pen']
open_textual_fields = ['processo',  'juiz', 'nome']

### auxiliary functions
# clean processo
def clean_processo(processo):
  # only numeric digits
  return re.sub(r'[^0-9]', '', processo)

# normalize text
def normalize_text(text, remove_spaces=True) -> str:
  text = str(text)
  
  # remove characters with accents (e.g. á, é, í) by converting them to ascii
  text = text.encode('ascii', 'ignore').decode('utf-8')
  
  if remove_spaces:
    text = text.replace(" ", "")
    # remove all non-alphanumeric characters after spaces removal
    text = ''.join(char for char in text if char.isalnum())
  else:
    # retain spaces if not explicitly removed; allow alphanumeric and space
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
  
  return text.upper()

### sccore functions
# numeric: difference between the two values

def score_numeric(result, expected, threshold = 0.1):
  try:
    # coerce to numeric because they might be strings
    result = pd.to_numeric(result, errors='coerce')
    expected = pd.to_numeric(expected, errors='coerce')

    # If conversion fails (NaN), return 0.
    if pd.isna(result):
        return 0
    
    diff = abs(result - expected)

    return 1 if diff <= threshold else 0
    
  except Exception:
    return 0

def parse_boolean(value):
  
  # if it's a boolen already, just return it back
  if isinstance(value, bool):
    return value
  
  if isinstance(value, str):
    # convert to lower case
    value = value.lower()
    
    # check if it's a boolean string
    if value == "true":
      return True
    
    elif value == "false":
      return False
    
    elif value in ["none", "null", ""]:
      return None

    else:
      # if it's not a boolean string, return None
      return 'error'
  

# boolearn
def score_boolean(result, expected):
  try:
    
    result = parse_boolean(result)
    expected = parse_boolean(expected)
    
    if result == 'error':
      return 0
    
    return 1 if result == expected else 0 
      
  except Exception: 
    return 0

# categorical
def score_categorical(result, expected):
  try:
    
    return 1 if normalize_text(result) == normalize_text(expected) else 0
    
  except Exception: 
    return 0
  
# open_textual
def score_open_textual(result, expected, remove_spaces=False):
  try:
    # remove spaces and normalize
    norm_result = normalize_text(result, remove_spaces=remove_spaces)
    norm_expected = normalize_text(expected, remove_spaces=remove_spaces)
    
    # check if the normalized result is equal to the normalized expected
    if norm_result == norm_expected:
      return 1
    
    else:
      # calculate the edit distance between the two strings
      similarity = SequenceMatcher(None, norm_result, norm_expected).ratio()
      
      # if the similarity is above 0.9, return 1
      # if the similarity is above 0.8, return 0.5
      # else return 0
      return 1 if similarity >= 0.9 else 0 #(0.5 if similarity >= 0.8 else 0)
    
  except Exception: 
    return 0
  
### evaluate each experiment
# function to receive the experiment, guess the data and evaluate

def evaluate_experiment(experiment, verbose=False):
  
  path = experiment
  name = re.sub(r"(_)?results_parsed\.parquet|experiment(_|[0-9])", "", experiment.split("/")[-1])

  ## load data
  # experiment
  try:
    df_experiment = pd.read_parquet(path)
    
  except Exception as e:
    print(f"  [Error] Experiment {name}: loading data: {e}")
    return

  # reference data
  df_reference = df_test.copy()
  
      
  # clean the processo column in both dataframes
  # if open_ai, pick th from the first row
  if "open_ai" in experiment:
    df_experiment['processo'] = df_experiment["custom_id"].apply(lambda x: x.split(":")[1])
  else:
    df_experiment["processo"] = df_experiment["processo"].apply(clean_processo)
    
  df_reference["processo"] = df_reference["processo"].apply(clean_processo)

  
  # check the sizes
  if df_experiment.shape[0] != df_reference.shape[0]:
    # retrieve only the rows in df_experiment that are also in df_reference
    df_experiment = df_experiment[df_experiment["processo"].isin(df_reference["processo"])]
    
    # check the sizes again
    if df_experiment.shape[0] != df_reference.shape[0]:
      print(f"  [Error] Experiment {name}: the number of rows in the experiment ({df_experiment.shape[0]}) is different from the reference ({df_reference.shape[0]})")
    # return {"score": [], "details": {}}
  
  # merge data
  try:
    # merge the dataframes on the processo column  
    merged_data = df_reference.merge(df_experiment, on='processo', suffixes=("_expected", "_result"))
    
  except Exception as e:
    print(f"  [Error] Experiment {name}: Error merging data: {e}")
    return {"score": [], "details": {}}

  # initialize lists to accumulate row scores and details
  all_row_scores = []
  all_row_details = []
  
  ### iterate over each line and each column
  for idx, row in merged_data.iterrows():
    
    row_total = 0 # total score for the row
    # row_count = 0 # total number of columns to be scored
    details = {} # details for the row
    
    # iterate over each column
    for col in df_reference.columns:
      
      # ignore 'processo' as it's the index row to mege both tables
      if col == "processo":
        continue
      
      # get the values
      try:
        result = row[f"{col}_result"]
        expected = row[f"{col}_expected"]
      except KeyError as e:
        print(f"  [Error] Experiment {name}: Error: missing key {e} in row {idx}")
        continue
      
      # define the score according with its type
      if col in numeric_fields:
        score = score_numeric(result, expected)
      elif col in boolean_fields:
        score = score_boolean(result, expected)
      elif col in categorical_fields:
        score = score_categorical(result, expected)
      elif col in open_textual_fields:
        score = score_open_textual(result, expected, remove_spaces=True)
      else:
        print(f"  [Error] Experiment {name}: Error: unknown column type for {col}")
        continue
      
      # add the score to the row total
      row_total += score
      # row_count += 1
      details[col] = score
      # print(f"  {col}: {score}")
      
    # calculate the row score
    row_score = row_total / (df_reference.shape[1] - 1) # divide by the number of columns to be scored, excluding 'processo'
    
    if verbose:
      print(f"Row {idx}: {row_score}")
      # print(f"  Result: {result}")
      # print(f"  Expected: {expected}")  
      # print(f"  Count: {row_count}")
      print(f"  Average: {row_score}")
      print(f"  Row: {row}")
      # print(f"  Result: {result}")
      # print(f"  Expected: {expected}")  
    # append the row score and details to the list
    all_row_scores.append(row_score)
    all_row_details.append(details)

  return {"score": all_row_scores, "details": all_row_details}  

  
## define an overall score to each experiment

overall_score = pd.DataFrame(columns=["experiment", "details", "score"])

#### iterate over each experiment
for experiment in experiments:
  print(f"Evaluating experiment {experiment}...")
  
  experiment_name = experiment.replace('../experiments/experiment_','').split('/')[0]
  score_details = evaluate_experiment(experiment)
  
  # new column
  new_row = pd.DataFrame({"experiment": [experiment_name], "details": [score_details["details"]], "score": [score_details["score"]]})
  overall_score = pd.concat([overall_score, new_row], ignore_index=True)
  
## calculate the overall score to each model
# overall_score["final_score"] = overall_score["score"].apply(lambda x: pd.Series(x).mean() if isinstance(x, list) else x)
# take care of consider null values as 0
overall_score["final_score"] = overall_score["score"].apply(lambda x: pd.Series(x).fillna(0).mean() if isinstance(x, list) else 0)


# function to compute the average score for each column in each experiment
def compute_avg_details(details):
  # if details is not a non-empty list, return empty dict
  if not isinstance(details, list) or len(details) == 0:
    return {}
  sums = {}
  counts = {}
  # each element in details is a dict for a row
  for row_detail in details:
    for col, score in row_detail.items():
      sums[col] = sums.get(col, 0) + score
      counts[col] = counts.get(col, 0) + 1
  # compute average for each column
  return {col: sums[col] / counts[col] for col in sums}


# to each experiment, calculate the average score for each column using the details field
overall_score["col_score"] = overall_score["details"].apply(compute_avg_details)

# now we want to check the average score for each column-type (to identify the performance in each kind of text)

df_column_result = pd.DataFrame(columns=["experiment", "column", "column_type", "score"])

for experiment in overall_score["experiment"]:
    
    name = experiment#.split("/")[-1].split(".")[0]
    
    # get the details for the experiment
    details = overall_score[overall_score["experiment"] == experiment]["col_score"].values[0]
    
    # iterate over each column
    for col, score in details.items():
        # get the column type
        if col in numeric_fields:
            col_type = "numeric"
        elif col in boolean_fields:
            col_type = "boolean"
        elif col in categorical_fields:
            col_type = "categorical"
        elif col in open_textual_fields:
            col_type = "open_textual"
        else:
            col_type = "unknown"
        
        # add the row to the dataframe
        new_row = pd.DataFrame({"experiment": [name], "column": [col], "column_type": [col_type], "score": [score]})
        df_column_result = pd.concat([df_column_result, new_row], ignore_index=True)  
  
df_column_result

# describing the results according with column type
result = (
  df_column_result.groupby(['experiment', 'column_type'])['score']
  # .fillna(0)
  .mean()
  .reset_index()
  .pivot(index='experiment', columns='column_type', values='score')
  .reset_index()
)
result = result[['experiment', 'numeric', 'boolean', 'categorical', 'open_textual']]
result

## calculate the final score for each experiment
# the final score is the average of all rows: i want a df with experiment | final_score
# final_score = pd.DataFrame(columns=["experiment", "final_score"])
# calculate the final score for each experiment by standardizing the experiment names
final_score = overall_score.copy()
# final_score["experiment"] = final_score["experiment"]#.apply(lambda x: x.split("/")[-1].split(".")[0] if isinstance(x, str) else x)
final_score = final_score[["experiment", "final_score"]]

# remove final_score null and order by experiment name
final_score = final_score[final_score["final_score"].notnull()]
final_score = final_score.sort_values(by="final_score", ascending=True)

# merge with values of result
final_score = final_score.merge(result, on="experiment")

# round results
final_score = final_score.round(3)

final_score

### calculate accuracy, precision, recall and f1 score
def compute_metrics_from_details(details, threshold=0.9):
    """
    Given details (list of dictionaries, each representing cell scores for a row),
    aggregate the scores and compute binary metrics.
    For ground truth, we assume a cell is correct (1) only if its score equals 1; otherwise 0.
    Prediction: a cell is predicted positive if its score is >= threshold.
    """
    cell_scores = []
    for row_detail in details:
        for score in row_detail.values():
            cell_scores.append(score)
    if not cell_scores:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
    y_true = [1 if score == 1 else 0 for score in cell_scores]
    y_pred = [1 if score >= threshold else 0 for score in cell_scores]
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    
    return metrics

# For each experiment, calculate the metrics from its cell-level details.
metrics_list = []

for idx, row in overall_score.iterrows():
    experiment_name = row["experiment"]
    details = row["details"]  # This is a list of dicts for each row.
    metrics = compute_metrics_from_details(details, threshold=0.9)
    metrics["experiment"] = experiment_name
    metrics_list.append(metrics)


metrics_df = pd.DataFrame(metrics_list)
metrics_df
  

# ### plot results
# import seaborn as sns
# import textwrap
# import matplotlib.pyplot as plt

# # Redefinir a figura com estilo visual aprimorado
# sns.set_theme(style="darkgrid", palette="pastel")
# fig, ax = plt.subplots(figsize=(14, 6))

# # Ocultar eixos
# ax.axis("off")

# # Criar uma tabela no gráfico com estilo
# table = plt.table(cellText=final_score.values,
#           colLabels=final_score.columns,
#           cellLoc='center',
#           loc='center',
#           colColours=["#003366"]*len(final_score.columns),
#           colWidths=[0.2]*len(final_score.columns))

# # Estilização da tabela
# table.auto_set_font_size(False)
# table.set_fontsize(10)
# table.scale(1, 2)

# # Aplicar quebra de linha (wrap) para o conteúdo da coluna 1 (índice 0)
# for key, cell in table.get_celld().items():
#   # key é uma tupla (linha, coluna); pulando o cabeçalho (linha 0)
#   if key[1] == 0 and key[0] > 0:
#     original_text = cell.get_text().get_text()
#     wrapped_text = "\n".join(textwrap.wrap(original_text, width=30))
#     cell.get_text().set_text(wrapped_text)

# # Cabeçalho com cor branca e negrito
# for i in range(len(final_score.columns)):
#   cell = table[0, i]
#   cell.set_text_props(color='white', weight='bold')

# # highlight_row = 5
# # Destaque para a linha do modelo fine-tuned
# # for i in range(len(final_score.columns)):
# #   table[(highlight_row+1, i)].set_facecolor('#b9f6ca')  # verde pastel

# # Título estilizado
# plt.title("Preliminary Results – OpenAI Model Comparisons", fontsize=16, weight='bold', color='#003366', pad=20)

# # Salvar imagem final
# plt.savefig("preliminary_results_table_stylish.png", dpi=300, bbox_inches='tight', transparent=True)
# plt.show()

# #### radar plot

# # Criar múltiplos gráficos de radar, um para cada experimento
# num_experiments = len(final_score)
# cols = 3
# rows = int(np.ceil(num_experiments / cols))

# fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), subplot_kw=dict(polar=True))
# axes = axes.flatten()

# # Define radar plot configuration variables
# exp_columns = ['numeric', 'boolean', 'categorical', 'open_textual']
# angles = np.linspace(0, 2 * np.pi, len(exp_columns), endpoint=False).tolist()
# angles += angles[:1]  # complete the loop for radar chart
# colors = plt.cm.viridis(np.linspace(0, 1, len(final_score)))
# labels = exp_columns

# for idx, row in final_score.iterrows():
#     values = row[exp_columns].tolist()
#     values += values[:1]

#     ax = axes[idx]
#     ax.plot(angles, values, color=colors[idx % len(colors)], linewidth=2)
#     ax.fill(angles, values, color=colors[idx % len(colors)], alpha=0.2)

#     ax.set_title(row["experiment"], fontsize=10, weight='bold', pad=10)
#     ax.set_theta_offset(np.pi / 2)
#     ax.set_theta_direction(-1)
#     ax.set_thetagrids(np.degrees(angles[:-1]), labels)
#     ax.set_ylim(0.0, 1.0)

# # Remover subplots não utilizados
# for j in range(idx + 1, len(axes)):
#     fig.delaxes(axes[j])

# plt.suptitle("Radar Charts – Individual Experiment Performance", fontsize=16, fontweight='bold')
# plt.tight_layout()
# plt.subplots_adjust(top=0.92)
# plt.savefig("radar_charts_per_experiment.png", dpi=600)
# plt.show()
