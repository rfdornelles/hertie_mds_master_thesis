import os
from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd
import matplotlib.pyplot as plt

# load environment variables
load_dotenv()
fine_tuning_job_id='ftjob-Cde9cY19khBG4TCRPjeClpAe'


# connect the client
client = OpenAI(
  api_key=os.getenv('HERTIE_OPENAI_API_KEY'),
  organization=os.getenv('HERTIE_OPENAI_ORG_ID'),
  project=os.getenv('HERTIE_OPENAI_PROJECT_ID')
)

# retrieve the fine-tuning job
response =  client.fine_tuning.jobs.retrieve(
    fine_tuning_job_id=fine_tuning_job_id
  )

# set a high limit to retrieve all events
evenets_response = client.fine_tuning.jobs.list_events(fine_tuning_job_id, limit= 9999).data
records = [e.data for e in events_response if e.data and len(e.data) > 0]

# convert to pandas dataframe
df = pd.DataFrame(records)

df.to_csv(f"metadata_fine_tuning_job_{fine_tuning_job_id}.csv", index=False)

df_plot = df[['step', 'train_loss', 'valid_loss']].copy()
df_plot.loc[:, 'epoc'] = df['step'] / 183

df_plot = df[['step', 'train_loss', 'valid_loss']].sort_values('step')

# 3) Fill NaNs so every step has a value
#    - forward fill: carry last known valid_loss forward
#    - backward fill: carry first known valid_loss backward (so you don't start with NaN)
df_plot[['valid_loss']] = (
    df_plot[['valid_loss']]
     .ffill()
     .bfill()
)

# 4) Plot
plt.figure(figsize=(8,5))
plt.plot(df_plot.step, df_plot.train_loss,      label='Train Loss')
plt.plot(df_plot.step, df_plot.valid_loss,      label='Valid Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()
plt.title('Fine-tune Loss Curves')
plt.savefig(f"fine_tune_loss_curves_{fine_tuning_job_id}.png", dpi=900)
plt.show()
