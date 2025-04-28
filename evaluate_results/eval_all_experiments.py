## Objective: evaluate all experiments results

import os
import json
import pandas as pd
from datasets import load_from_disk
from difflib import SequenceMatcher
import numpy as np
import re
from sklearn.metrics import accuracy_score #, precision_score, recall_score, f1_score


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
open_textual_fields = ['juiz', 'nome']

## evaluation according with the NLP task
ner_fields = [
    "juiz",      # judge's full name
    "nome",      # defendant's full name
    "pena_base", # e.g. "8a" / "9a 6m"
    "tot_pen"    # e.g. "10a 2m"
]

# Numeric QA fields (number + unit)
numeric_qa_fields = [
    "maconha_g", "cocaina_g", "crack_g", "ecstasy_g", "lsd_g",
    "tot_pen_meses"
]

# Boolean flags split into two modeling paradigms:
classification_boolean = [
    # judicial evaluations
    'aval_antecedentes','aval_conduta','aval_personalidade','aval_natureza',
    'aval_quantidade','aval_variedade','aval_circunstancias','aval_consequencias','aval_culpabilidade',
    # procedural outcomes
    'resultado_art_28','resultado_art_33','resultado_art_34','resultado_art_35',
    'denuncia_art_33','denuncia_art_34','denuncia_art_35'
]
qa_boolean = [
    # drug‐seizure facts
    'maconha','maconha_outras','cocaina','crack','ecstasy','lsd','outras_drogas',
    # situational facts
    'flag_local_de_trafico','flag_preso_no_momento_da_sentenca',
    'flag_confissao_informal','flag_confissao',
    'flag_denuncia_anonima','flag_denuncia',
    'flag_atitude_suspeita','flag_divergencias_nos_relatos_dos_policiais',
    'flag_investigacao','flag_interceptacao','flag_mandado'
]

# Multiclass classification fields
multiclass_classification_fields = [
    "local",     # 9 possible locais
    "sentenca"   # {“Absolvição”,“Desclassificação”,“Parcialmente Procedente”,“Procedente”}
]

# Open‐text normalization fields
open_textual_fields = ['juiz', 'nome']

# Build the canonical NLP‐task mapping
nlp_task_fields = {
    "named_entity_recognition": ner_fields,
    "numeric_qa":               numeric_qa_fields,
    "binary_classification":    classification_boolean,
    "yesno_qa":                 qa_boolean,
    "multiclass_classification":multiclass_classification_fields,
    "open_text_normalization":  open_textual_fields
}

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

def evaluate_experiment(experiment):
  
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
      print(f"  [Alert!] Experiment {name}: the number of rows in the experiment ({df_experiment.shape[0]}) is different from the reference ({df_reference.shape[0]})")
  
  # merge data
  try:
    # merge the dataframes on the processo column  
    merged_data = df_reference.merge(df_experiment, how = 'left', on='processo', suffixes=("_expected", "_result"))
    
  except Exception as e:
    print(f"  [Error] Experiment {name}: Error merging data: {e}")
    return {"score": [], "details": {}}

  # initialize lists to accumulate row scores and details
  all_row_details = []
  
  ### iterate over each line and each column
  for idx, row in merged_data.iterrows():
  
    details = {} # details for the row
    
    # iterate over each column
    for col in df_reference.columns:
      
      # ignore 'processo' as it's the index row to mege both tables
      if col == "processo":
        continue
      
      # get the values
      try:
        details["experiment"] = name
        
        expected = row[f"{col}_expected"]
        result = row[f"{col}_result"] if f"{col}_result" in row and pd.notna(row[f"{col}_result"]) else None
        
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
      
      # row_count += 1
      details[col] = score
      # print(f"  {col}: {score}")

      
    all_row_details.append(details)

    # return a dataframe
    try:
      result = pd.DataFrame(all_row_details) 
      # result = pd.concat([pd.DataFrame({'experiment': name}), all_row_details], axis=1) 
      
    except Exception as e:
      print(f"  [Error] Experiment {name}: Error creating dataframe: {e}")
      return None
    
  return result

  
## define an overall score to each experiment

df_experiments_score = pd.DataFrame()

#### iterate over each experiment
for experiment in experiments:
  print(f"Evaluating experiment {experiment}...")
  
  experiment_name = experiment.replace('../experiments/experiment_','').split('/')[0]
  score_details = evaluate_experiment(experiment)
  
  # add the rows to the dataframe
  if score_details is not None:
    # add the experiment name to the dataframe
    score_details["experiment"] = experiment_name
    # add the rows to the dataframe
    df_experiments_score = pd.concat([df_experiments_score, score_details], ignore_index=True)
  else:
    print(f"  [Error] Experiment {experiment}: Error evaluating experiment")
    continue

  
 ## calculating the overall score
 
 ## remove the id columns
eval_cols = [col for col in df_experiments_score.columns if col not in ["experiment", "processo"]]

metrics_list = []

for experiment, group in df_experiments_score.groupby("experiment"):
   
  # calculate the score for each column
  y_pred = group[eval_cols].values.flatten()
  # the ground_true will be 1 always -- since it was already evaluated
  y_true = np.ones_like(y_pred)
  
  ## calculate metrics
  acc = accuracy_score(y_true, y_pred)
  metrics = {"experiment": experiment, "accuracy": acc}
  metrics_list.append(metrics)
  

df_all_metrics = pd.DataFrame(metrics_list)

### calculate the score (accuracy and f1) each model has in each task

task_fields = {
    "numeric": numeric_fields,
    "boolean": boolean_fields,
    "categorical": categorical_fields,
    "open_textual": open_textual_fields
}

df_task_metrics = pd.DataFrame(columns=["experiment", "column_type", "score_accuracy"])

task_metrics_list = []

for experiment, group in df_experiments_score.groupby("experiment"):
    
    # for each task type
    for task_type, expected_fields in task_fields.items():
      
      # filter the group by task type
      pred_cols = [field for field in expected_fields if field in group.columns]
      gt_cols = [f"gt_{field}" for field in pred_cols if f"gt_{field}" in group.columns]
      
      y_pred = group[pred_cols].to_numpy().flatten()
      y_true = np.ones_like(y_pred)

      acc = accuracy_score(y_true, y_pred)
          
      task_metrics_list.append({
        "experiment": experiment,
        "column_type": task_type,
        "score_accuracy": acc
    })
         
df_task_metrics = pd.DataFrame(task_metrics_list)
df_task_metrics
    
## pivot the results
df_tasks_metrics_accuracy = df_task_metrics.pivot(index="experiment", columns="column_type", values="score_accuracy").reset_index()
    

## add the overall metrics to the tasks metrics
df_tasks_metrics_accuracy = df_tasks_metrics_accuracy.merge(
  df_all_metrics[['experiment', 'accuracy']], on="experiment", how="left"
).rename(columns={"accuracy": "overall_accuracy"})

# Reorder columns: place overall_accuracy as the first column after experiment
cols = df_tasks_metrics_accuracy.columns.tolist()
cols.remove("experiment")
cols.remove("overall_accuracy")
df_tasks_metrics_accuracy = df_tasks_metrics_accuracy[["experiment", "overall_accuracy"] + cols]

# Order the dataframe by overall_accuracy in descending order
df_tasks_metrics_accuracy = df_tasks_metrics_accuracy.sort_values("overall_accuracy", ascending=False)

####### calculate also the NLP task-based metrics

df_nlp_task_metrics = pd.DataFrame(columns=[
    "experiment", "column_type",
    "accuracy", "f1_macro"
])

nlp_task_metrics_list = []

for experiment, group in df_experiments_score.groupby("experiment"):
  
    # for each task type
    for task_type, expected_fields in nlp_task_fields.items():
      
      # filter the group by task type
      pred_cols = [field for field in expected_fields if field in group.columns]
      gt_cols = [f"gt_{field}" for field in pred_cols if f"gt_{field}" in group.columns]
      
      y_pred = group[pred_cols].to_numpy().flatten()
      y_true = np.ones_like(y_pred)

      acc = accuracy_score(y_true, y_pred)
          
      nlp_task_metrics_list.append({
        "experiment": experiment,
        "column_type": task_type,
        "score_accuracy": acc
    })

df_nlp_task_metrics = pd.DataFrame(nlp_task_metrics_list)
# pivot
df_nlp_task_metrics_accuracy = df_nlp_task_metrics.pivot(index="experiment", columns="column_type", values="score_accuracy").reset_index()

# add the overall metrics to the tasks metrics
df_nlp_task_metrics_accuracy = df_nlp_task_metrics_accuracy.merge(
  df_all_metrics[['experiment', 'accuracy']], on="experiment", how="left"
).rename(columns={"accuracy": "overall_accuracy"})
 
 
######## save results
df_tasks_metrics_accuracy.to_csv("tasks_metrics_accuracy.csv", index=False)
# df_tasks_metrics_f1.to_csv("tasks_metrics_f1.csv", index=False)
df_experiments_score.to_csv("experiments_score.csv", index=False)
df_all_metrics.to_csv("all_metrics.csv", index=False)
df_nlp_task_metrics_accuracy.to_csv("nlp_tasks_metrics_accuracy.csv", index=False)
 
