###### sample
set.seed(2025)

# read files
df_2024 <- arrow::read_parquet("data/clean_tjsp_drug_cases_2024.parquet") |> 
  dplyr::mutate(ano = '2024') 
df_2025 <- arrow::read_parquet("data/clean_tjsp_drug_cases_2025.parquet") |> 
  dplyr::mutate(ano = '2025')

# remove the too old
df_2024 <- df_2024 |> 
  dplyr::filter(disponibilizacao <= max(df_2025$disponibilizacao))

# merge
df_drug_cases <- dplyr::bind_rows(df_2024, df_2025)

# remove potential duples
df_drug_cases <- df_drug_cases |> 
  dplyr::anti_join(df_drug_cases |> 
  janitor::get_dupes(processo))

df_drug_cases |> 
  janitor::get_dupes(processo)

## estimating the size

n = nrow(df_drug_cases)

# error margin
e = 0.05

# confidence level
confidence = 0.95

# proporcion 
p = 0.5

sample_size <- sampler::rsampcalc(
  N = n,
  e = e*100,
  ci = confidence*100,
  p = p
)

print(sample_size)

# adding some margin
sample_size <- round(sample_size*1.2)

## sampling
# df_drug_cases |>
#   sampler::ssamp(
#     n = sample_size,
#     strata = ano
# )  |>
#   dplyr::count(ano)

## get 200 cases from each group
df_drug_cases_sample <- df_drug_cases |> 
  dplyr::group_by(ano) |> 
  dplyr::slice_sample(n = sample_size/2) |> 
  dplyr::ungroup() |> 
  dplyr::select(-sentence_lenght)

df_drug_cases_sample |> 
  dplyr::count(ano)

arrow::write_parquet(df_drug_cases, "data/sample_454_tjsp_drug_cases_2024-2025_v1.parquet")

