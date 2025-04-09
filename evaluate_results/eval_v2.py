## Objective: evaluate all experiments results
import os
import json
import pandas as pd
from datasets import load_from_disk
from difflib import SequenceMatcher

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
      
      # add the experiment full path to the list
      experiments.append(full_path)
      print(f"  Experiment: {full_path}")

### load the validation data
print("Loading validation data...")
df_validation = load_from_disk("../data/validation").to_pandas().pop("output").apply(json.loads).apply(pd.Series)
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

def score_numeric(result, expected):
  try:
    diff = abs(float(result) - float(expected))

    return 1 if diff == 0 else 0
    
  except Exception:
    return 0


# boolearn
def score_boolean(result, expected):
  try:
    test = bool(result) == bool(expected)
    return 1 if test else 0    
      
  except Exception: 
    return 0

# categorical
def score_categorical(result, expected):
  try:
    
    test = normalize_text(result) == normalize_text(expected)
    
    return 1 if test else 0
    
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
      return 1 if similarity >= 0.9 else (0.5 if similarity >= 0.8 else 0)
    
  except Exception: 
    return 0
  
### evaluate each experiment
# function to receive the experiment, guess the data and evaluate

def evaluate_experiment(experiment, verbose=False):
  
  path = experiment
  name = experiment.split("/")[-1].split(".")[0]

  ## load data
  # experiment
  try:
    df_experiment = pd.read_parquet(path)
    
  except Exception as e:
    print(f"Error loading data: {e}")
    return

  # reference data
  df_reference = df_validation.copy() if name.__contains__("validation") else df_test.copy()
  
  # check the sizes
  if df_experiment.shape[0] != df_reference.shape[0]:
    print(f"Error: the number of rows in the experiment ({df_experiment.shape[0]}) is different from the reference ({df_reference.shape[0]})")
    return {"score": [], "details": {}}
  
  # merge data
  try:
    merged_data = df_reference.merge(df_experiment, on='processo', suffixes=("_result", "_expected"))
    
  except Exception as e:
    print(f"Error merging data: {e}")
    return {"score": [], "details": {}}

  # initialize lists to accumulate row scores and details
  all_row_scores = []
  all_row_details = []
  
  ### iterate over each line and each column
  for idx, row in merged_data.iterrows():
    
    row_total = 0 # total score for the row
    row_count = 0 # total number of columns to be scored
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
        print(f"Error: missing key {e} in row {idx}")
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
        print(f"Error: unknown column type for {col}")
        continue
      
      # add the score to the row total
      row_total += score
      row_count += 1
      details[col] = score
      # print(f"  {col}: {score}")
      
    # calculate the row score
    row_score = row_total / row_count if row_count > 0 else 0
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

overall_score = pd.DataFrame(columns=["experiment", "score", "details"])

#### iterate over each experiment
for experiment in experiments:
  print(f"Evaluating experiment {experiment}...")
  
  score_details = evaluate_experiment(experiment, verbose=True)
  
  overall_score = pd.concat([overall_score, pd.DataFrame({"experiment": experiment, "score": score_details["score"], "details": score_details["details"]})], ignore_index=True)
  
# calculate the overall score to each model
overall_score["score"] = overall_score["score"].apply(lambda x: x.mean() if isinstance(x, list) else x)
overall_score["details"] = overall_score["details"].apply(lambda x: x.mean() if isinstance(x, list) else x)

# save the overall score to a files  
  