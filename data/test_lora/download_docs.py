# download data from bq and transform into Huggingface dataset format
from google.cloud import bigquery
import pandas as pd
import os
import json
from datasets import Dataset

query = """
SELECT DISTINCT processo, cd_doc, julgado FROM `rfdornelles-bq.tjsp.tjsp_sentencas_2023` 

WHERE ARRAY_LENGTH(SPLIT(julgado, ' ')) >= 6000
"""

client = bigquery.Client(project='rfdornelles-bq')
df = client.query(query).to_dataframe()

hf_dataset = Dataset.from_pandas(df)

hf_dataset.save_to_disk('data/test_lora')