## objective: run trough the OpenAI api the validation samples
# using the chepeast model available (4o-mini)

# reference: https://cookbook.openai.com/examples/batch_processing

import dotenv
import pandas as pd
import os
from openai import OpenAI
import json
from datasets import load_from_disk
import time
from datetime import datetime


#### load environment variables from .env file
dotenv.load_dotenv()

API_KEY = os.getenv("HERTIE_OPENAI_API_KEY")

#### definitions
model = 'gpt-4o-mini'
experiment = os.path.basename(os.path.abspath(os.path.dirname(__file__)))
dataset = "test"
prompt = "prompt_v4_clean.md"
schema = "schema.json"


print(f"Running experiment: {experiment}")
print(f"Using model: {model}")
print(f"Using dataset: {dataset}")
print(f"Using prompt: {prompt}")
print(f"Using schema: {schema}")
#### download data
df = load_from_disk(f"../../data/{dataset}").to_pandas()

### load prompt
with open(f"../../prompt/{prompt}", "r") as f:
    prompt = f.read()
    
### load schema
schema = json.load(open(f"../../prompt/{schema}", "r"))

### build the client
client = OpenAI(api_key=API_KEY)

## generate the batch
# check if the json file already exists
if os.path.exists(f"{experiment}-batch.jsonl"):
    print(f"Batch file {experiment}-batch.jsonl already exists.")
  
else:
  print(f"Batch file {experiment}-batch.jsonl does not exist. Generating a new one.")
  tasks = []

  for index, row in df.iterrows():
    
      description = f"{experiment}:{row['processo']}:{json.loads(row['output'])['nome']}"
      task = {
          "custom_id": description,
          "method": "POST",
          "url": "/v1/chat/completions",
          "body": {
              "model": model,
              "temperature": 0,
              "max_tokens": 1000,
              "messages": [
                  { "role": "system", "content": prompt },
                  { "role": "system", "content": f"O número do processo deve ser: {json.dumps(row['processo'])}"},
                  { "role": "user", "content": f"SENTENÇA/ATA A SER ANALISADA:     {json.dumps(row['input'])}" }
              ],
              "response_format": { 
                  "type": "json_schema", 
                  "json_schema": schema 
              }
          }


      }
      tasks.append(task)
      
  ## generate the .jsonl file
  file_name = f"{experiment}-batch.jsonl"
  with open(file_name, "w") as f:
      for task in tasks:
          f.write(json.dumps(task) + "\n")
  print(f"Batch file {file_name} generated with {len(tasks)} tasks.")

  ## upload the batch
  batch_file = client.files.create(
      file = open(file_name, "rb"),
      purpose="batch"
  )
  print(f"Batch file {batch_file.id} uploaded.")

  ## create the batch job

  batch_job = client.batches.create(
    input_file_id= batch_file.id,
    endpoint="/v1/chat/completions",
    completion_window='24h'
  )

  # save the batch job id
  with open(f"{experiment}-batch_job_id.txt", "w") as f:
      f.write(batch_job.id)
  print(f"Batch job {batch_job.id} created.")

# get the batch job id
with open(f"{experiment}-batch_job_id.txt", "r") as f:
    batch_job_id = f.read()

## check status
def get_batch_job_status(batch_job_id):
    batch_job_status = client.batches.retrieve(batch_job_id)
    
    if (batch_job_status.status == "processing"):
        print(f"Batch job is still processing: {batch_job_status.status}")
        return batch_job_status
    elif (batch_job_status.status == "in_progress"):
        print(f"Batch job is in progress: {batch_job_status.status}")
        return batch_job_status
    elif (batch_job_status.status == "completed"):
        print(f"Batch job completed: {batch_job_status.status}")
        # download the results
        print(f"Completed!!")
        print(f"Completed at: {datetime.fromtimestamp(batch_job_status.completed_at)}")
        return batch_job_status
    elif (batch_job_status.status == "failed"):
        print(f"Batch job failed: {batch_job_status.status}")

    return batch_job_status

status = get_batch_job_status(batch_job_id) 

while status.status not in ["completed", "failed"]:
    print("Waiting for batch job to complete...")
    print(f"Status: {status.status}")
    print(f"Job started at: {datetime.fromtimestamp(status.created_at)}")
    print(f"Job expiration: {datetime.fromtimestamp(status.expires_at)}")
  
    
    total = status.request_counts.total
    completed = status.request_counts.completed
    failed = status.request_counts.failed
    
    if total > 0:
        print(f"Total requests: {total} - Completed: {completed} ({completed/total:.2f}%) - Failed: {failed} ({failed/total:.2f}%)")
    else:
        print(f"Total requests: {total} - Completed: {completed} - Failed: {failed}")

 
    
    time.sleep(60*30)
    status = get_batch_job_status(batch_job_id)

output_path = f"{experiment}-batch_results.jsonl"

if os.path.exists(output_path):
    print(f"Batch results file {output_path} already exists.")
    
else:
    print(f"Batch results file {output_path} does not exist. Generating a new one.")
    # download the results
    result_file_id = status.output_file_id
    result = client.files.content(result_file_id).content
    
    # write the result to a file
    try: 
      with open(output_path, "wb") as f:
        f.write(result)
    except Exception as e:
        print(f"Error writing file: {e}")
        exit(1)
  
    print(f"Batch job results saved to {output_path}")

## open the results
with open(output_path, "r") as f:
    lines = f.readlines()
    results = []
    for line in lines:
        try:
            result = json.loads(line)
            results.append(result)
        except Exception as e:
            print(f"Error parsing line: {e}")
            print(line)
            continue

    df_results = pd.DataFrame(results)
    df_results.to_csv(f"{experiment}-batch_metadata_results.csv", index=False)
    print(f"Batch job results saved to {experiment}-batch_results.csv")

## process the results

df_rows = []
for index, row in df_results.iterrows():
  try:
    # load the content from the response
    content = json.loads(row['response']['body']['choices'][0]['message']['content'])
    # capture custom_id and set it as the first key in a new dictionary
    custom_id = row.get('custom_id', None)
    new_content = {'custom_id': custom_id}
    # merge the rest of the content so that custom_id remains the first column
    new_content.update(content)
    df_rows.append(new_content)
    
  except Exception as e:
    print(f"Error parsing line: {e}")
    print(row)
    continue
df_rows = pd.DataFrame(df_rows)

df_rows

## save
df_rows.to_parquet(f"{experiment}-batch_results_parsed.parquet", index=False)

print(f"Batch job results parsed and saved to {experiment}-batch_results_parsed.parquet")