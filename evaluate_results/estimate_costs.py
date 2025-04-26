import pandas as pd
import tiktoken
import os
from datetime import timedelta

# ---- 1) Token-based cost for OpenAI GPT-4o-mini FT ------------------------

# Paths and params
parquet_path       = "../data/sample_454_tjsp_drug_cases_2024-2025_v1.parquet"
prompt_path        = "../prompt/prompt_v4.md"
experiment_folder  = "../tjsp_experiment/tjsp_2024_2025_sample_454"
batch_size         = 454
cost_input_per_m   = 0.13   # €/million tokens
cost_output_per_m  = 0.53   # €/million tokens

# Try tiktoken, fallback to whitespace tokens
try:
    import tiktoken
    enc = tiktoken.encoding_for_model("gpt-4")
    tokenize = lambda txt: len(enc.encode(txt))
except ImportError:
    print("tiktoken not installed; using whitespace split as token approx.")
    tokenize = lambda txt: len(str(txt).split())

# Load data and compute input tokens
df = pd.read_parquet(parquet_path)
df["tok_data"]  = df["julgado"].apply(tokenize)
total_tokens    = df["tok_data"].sum()
prompt_text     = open(prompt_path, "r", encoding="utf-8").read()
prompt_tokens   = tokenize(prompt_text)
df["tok_total"] = df["tok_data"] + prompt_tokens
total_tokens_all = df["tok_total"].sum()

# Compute output tokens
output_tokens = 0
for fn in os.listdir(experiment_folder):
    if fn.endswith(".json"):
        content = open(os.path.join(experiment_folder, fn), "r", encoding="utf-8").read()
        output_tokens += tokenize(content)

gpt_cost_total      = (total_tokens_all/1e6)*cost_input_per_m + (output_tokens/1e6)*cost_output_per_m
gpt_cost_per_sample = gpt_cost_total / batch_size

# ---- 2) Energy-based cost for open-source models -------------------------

csv_paths = {
    "Phi-4":        "phi_4_costs.csv",
    "Llama 3.2-3B": "llama_3.2_3b_costs.csv"
}
# EU price: https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Electricity_price_statistics#Electricity_prices_for_household_consumers
tariff_kwh = 0.2889 #0.3951  # €/kWh (example)

records = []
for model, path in csv_paths.items():
    df_log = pd.read_csv(path)
    # compute total time in seconds and formatted string
    total_sec = df_log["Relative Time (Process)"].max()
    time_str  = str(timedelta(seconds=int(round(total_sec))))
    # sum peak power of all GPUs per instant
    max_cols = [c for c in df_log.columns if ".powerWatts__MAX" in c]
    df_log["p_tot_w"] = df_log[max_cols].sum(axis=1)
    df_log = df_log.sort_values("Relative Time (Process)").reset_index(drop=True)
    df_log["dt_s"] = df_log["Relative Time (Process)"].diff().fillna(0)
    e_ws = (df_log["p_tot_w"] * df_log["dt_s"]).sum()
    e_kwh = e_ws / 3.6e6
    cost_total = e_kwh * tariff_kwh
    e_wh_samp  = e_kwh * 1000 / batch_size
    cost_samp  = cost_total / batch_size
    time_per_sample = total_sec / batch_size

    records.append({
        "Model":              model,
        "Total Time":         time_str,
        "Time/Sample (s)":    round(time_per_sample, 2),
        "Total Energy (kWh)": round(e_kwh, 3),
        "Total Cost (€)":     round(cost_total, 2),
        "Energy/Sample (Wh)": round(e_wh_samp, 2),
        # "Cost/Sample (€)":    round(cost_samp, 4)
    })

# Add GPT row
records.append({
    "Model":              "GPT-4o-mini FT",
    "Total Time":         "unknown",
    "Time/Sample (s)":    "unknown",
    "Total Energy (kWh)": "unknown",
    "Total Cost (€)":     round(gpt_cost_total, 2),
    "Energy/Sample (Wh)": "unknown",
    # "Cost/Sample (€)":    round(gpt_cost_per_sample, 4)
})

# create summary DataFrame
df_summary = pd.DataFrame(records)
df_summary
df_summary.to_csv("summary_costs.csv", index=False)
