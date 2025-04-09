## objective: generate batches to be finetuned by openai
# refs:
  # https://platform.openai.com/docs/guides/fine-tuning
  # https://cookbook.openai.com/examples/how_to_finetune_chat_models
  # https://cookbook.openai.com/examples/chat_finetuning_data_prep
  
import dotenv
import os
from openai import OpenAI
from datasets import load_from_disk
import json
from datetime import datetime
import pandas as pd
import time

## load environment variables
dotenv.load_dotenv()
openai_api_key = os.getenv("HERTIE_OPENAI_API_KEY")

## model
model = "gpt-4o-mini-2024-07-18"
df_train = load_from_disk('../data/train/').to_pandas()

df_test = pd.concat([load_from_disk('../data/test/').to_pandas(), load_from_disk('../data/validation/').to_pandas()])

experiment = f"fine_tuning_{model}"

## prompt 
with open(f"../prompt/prompt_v4_clean.md", "r") as f:
    prompt = f.read()

### load schema
schema = json.load(open(f"../prompt/schema.json", "r"))

## generate the batch
# check if the json file already exists
if os.path.exists(f"{experiment}-batch_train.jsonl") and os.path.exists(f"{experiment}-batch_test.jsonl"):
    print(f"Batch file {experiment}-batch_train.jsonl and {experiment}-batch_test.jsonl already exists.")
  
else:
  # generating training data
  print(f"Batch file {experiment}-batch.jsonl does not exist. Generating a new one.")
  training_tasks = []
  
  ## removing cases that are too large
  too_large = ['00120876720168260635', '00013583920178260540', '00689253920168260050', '00487303320168260050', '00126531620168260635']

  for index, row in df_train.iterrows():
    
      if row['processo'] in too_large:
          continue
      
      task = {"messages": [
                  { "role": "system", "content": f"{prompt}\n\n" + f"O número do processo deve ser: {json.dumps(row['processo'])}"},
                  { "role": "user", "content": f"SENTENÇA/ATA A SER ANALISADA:     {json.dumps(row['input'])}" },
                  { "role": "assistant", "content": f"{json.dumps(row['output'])}" }
      ]}                    
      training_tasks.append(task)
      
  ## generate the .jsonl file
  file_name = f"{experiment}-batch_train.jsonl"
  with open(file_name, "w") as f:
      for task in training_tasks:
          f.write(json.dumps(task) + "\n")
  print(f"Batch training file {file_name} generated with {len(training_tasks)} tasks.")
  
  # generating test data
  test_tasks = []
  
  for index, row in df_test.iterrows():
  
      if row['processo'] in too_large:
       continue
        
    
      task = {"messages": [
                  { "role": "system", "content": f"{prompt}\n\n" + f"O número do processo deve ser: {json.dumps(row['processo'])}"},
                  { "role": "user", "content": f"SENTENÇA/ATA A SER ANALISADA:     {json.dumps(row['input'])}" },
                  { "role": "assistant", "content": f"{json.dumps(row['output'])}" }
      ]}                    
      test_tasks.append(task)

  ## generate the .jsonl file
  file_name = f"{experiment}-batch_test.jsonl"
  with open(file_name, "w") as f:
      for task in test_tasks:
          f.write(json.dumps(task) + "\n")
  print(f"Batch test file {file_name} generated with {len(test_tasks)} tasks.")

## init client
client = OpenAI(
    api_key=openai_api_key,
    organization="org-palfxcxiIQmDnuc12mwth2Lk",
    project = 'proj_3EVN3vioaKE2pssDhnpgPwL9'
)

# check if the files already exist
if os.path.exists("training_file_id.txt") and os.path.exists("validation_file_id.txt"):
    print("Training and validation file IDs already exist.")
    with open("training_file_id.txt", "r") as log_file:
        training_file_id = log_file.read()
    with open("validation_file_id.txt", "r") as log_file:
        validation_file_id = log_file.read()
else:
    print("Training and validation file IDs do not exist. Generating new ones.")  
  

    ## upload the training file
    def upload_file(file_name: str, purpose: str) -> str:
        with open(file_name, "rb") as file_fd:
            response = client.files.create(file=file_fd, purpose=purpose)
        return response.id


    training_file_id = upload_file(f"{experiment}-batch_train.jsonl", "fine-tune")
    validation_file_id = upload_file(f"{experiment}-batch_test.jsonl", "fine-tune")

    print("Training file ID:", training_file_id)
    print("Validation file ID:", validation_file_id)

    # write the ids
    with open("training_file_id.txt", "a") as log_file:
        log_file.write(training_file_id)

    with open("validation_file_id.txt", "a") as log_file:
        log_file.write(validation_file_id)

## check if the ft job already exists
if os.path.exists("fine_tuning_job_id.txt"):
    print("Fine-tuning job ID already exists.")
    with open("fine_tuning_job_id.txt", "r") as log_file:
        job_id = log_file.read()
else:
## create the fine-tune job
  response = client.fine_tuning.jobs.create(
    training_file=training_file_id,
    validation_file=validation_file_id,
    model=model,
    suffix="judicial_sentences",
    seed=77
  )

  job_id = response.id
  print("Fine-tuning job ID:", job_id)
  with open("fine_tuning_job_id.txt", "a") as log_file:
      log_file.write(job_id)
      

## retrive the fine-tune job
# check status
def get_batch_job_status(job_id):
    batch_job_status = client.fine_tuning.jobs.retrieve(job_id)
    
    if (batch_job_status.status == "running"):
        print(f"Batch job is still processing: {batch_job_status.status}")
        print(f"Estimated finish: {datetime.fromtimestamp(status.estimated_finish)}")
        return batch_job_status, 1
    elif (batch_job_status.status == "in_progress"):
        print(f"Batch job is in progress: {batch_job_status.status}")
        return batch_job_status, 0.5
    elif (batch_job_status.status == "completed"):
        print(f"Batch job completed: {batch_job_status.status}")
        # download the results
        print(f"Completed!!")
        print(f"Completed at: {datetime.fromtimestamp(batch_job_status.completed_at)}")
        return batch_job_status
    elif (batch_job_status.status == "failed"):
        print(f"Batch job failed: {batch_job_status.status}")
        print(f"Error code: {batch_job_status.error.code}")
        print(f"Error message: {batch_job_status.error.message}")

    return batch_job_status, 0

status, time_multiplier = get_batch_job_status(job_id) 

while status.status not in ["completed", "failed", "succeeded"]:
    print("Waiting for batch job to complete...")
    print(f"Status: {status.status}")
    print(f"Job started at: {datetime.fromtimestamp(status.created_at)}")
    # print(f"Job expiration: {datetime.fromtimestamp(status.expires_at)}")
    
    response = client.fine_tuning.jobs.list_events(job_id)

    events = response.data
    events.reverse()

    for event in events:
        print(event.message)
        
    
    time.sleep(60*time_multiplier) # wait for 5 minutes
    status, time_multiplier = get_batch_job_status(job_id)
    

response = client.fine_tuning.jobs.retrieve(job_id)
fine_tuned_model_id = response.fine_tuned_model

if fine_tuned_model_id is None:
    raise RuntimeError(
        "Fine-tuned model ID not found. Your job has likely not been completed yet."
    )

print("Fine-tuned model ID:", fine_tuned_model_id)

#### get the metrics
model_metrics = client.fine_tuning.jobs.list_events(job_id).data

# itera
model_metrics = [event.data for event in model_metrics if event.type == "metrics"]

pd.DataFrame(model_metrics).sort_values(by="step", ascending=True)