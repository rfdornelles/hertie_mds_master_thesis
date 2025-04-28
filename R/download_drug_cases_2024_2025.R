## download drug cases from São Paulo Tribunal de Justiça

# directory
dir = 'data/raw/tj_sp_drugs_2025'
dir.create(dir, recursive = TRUE, showWarnings = FALSE)

# pick the topics related to drug cases
assunto <- "3607,5885,6317,6318,3608,5894,5895,5896,5897,5898,5899,5900,5901"

run_code <- FALSE

if (run_code) { 
  tjsp::tjsp_baixar_cjpg(
    diretorio = dir,
    assunto = assunto,
    inicio = "01/01/2025",
    fim = "24/04/2025"
  )
}

## load data

# raw file
raw_file <- 'data/raw_tjsp_drug_cases_2025.parquet'

if (!file.exists(raw_file)) {

df_drug_cases <- tjsp::tjsp_ler_cjpg(
  diretorio = dir
)

# save
arrow::write_parquet(df_drug_cases, 
                     "data/raw_tjsp_drug_cases_2025.parquet")
}

df_drug_cases <- arrow::read_parquet(raw_file)


# assuming that the cut would be: 
# https://portal.stf.jus.br/processos/detalhe.asp?incidente=4034145
# publication: 1 day after 27/09/2024
# decison: 26/06/2024


# before_decision <-  lubridate::dmy("26/06/2024") - 30 # before 
# after_decision <-  lubridate::dmy("27/09/2024") + 30 #after (1 day in margin)

df_drug_cases_clean <- df_drug_cases |> 
  dplyr::filter(!is.na(julgado)) |> 
  # define the group
  dplyr::mutate(
    # group = dplyr::case_when(
    #   # disponibilizacao <= before_decision ~ "before_decision",
    #   # disponibilizacao >= after_decision ~ "after_decision",
    #   TRUE ~ "limbo"
    # ),
    sentence_lenght = stringr::str_length(julgado)
  ) |> 
  dplyr::filter(sentence_lenght > 2000)

# count
# df_drug_cases_clean |> 
  # dplyr::count(group)


# remove limbo
df_drug_cases_clean <- df_drug_cases_clean |> 
  # dplyr::filter(group != "limbo") |> 
  dplyr::select(-pagina:-duplicado)

# save the clean
arrow::write_parquet(df_drug_cases_clean, 
                     "data/clean_tjsp_drug_cases_2025.parquet")

df_drug_cases_clean |> 
  dplyr::count(processo, sort = TRUE)

df_drug_cases_clean |> 
  dplyr::count(processo, julgado, disponibilizacao, sort = TRUE) |> 
  dplyr::filter(n > 1)

df_drug_cases_clean |>  
  # dplyr::select(-cd_doc, duplicado, -hora_coleta) |> 
  janitor::get_dupes()

# df_drug_cases_clean |> 
#   dplyr::distinct(processo, disponibilizacao, .keep_all = TRUE) |> 
#   dplyr::group_by(processo) |> 
#   dplyr::filter(disponibilizacao == min(disponibilizacao)) |> 
#   dplyr::filter(sentence_lenght == max(sentence_lenght)) |>
#   dplyr::ungroup() |> 
#   dplyr::select(-cd_doc) |> 
#   dplyr::distinct() |> 
#   dplyr::arrange(disponibilizacao)  |> 
#   dplyr::count(processo, sort = TRUE) |>
#   dplyr::filter(n > 1)

df_drug_cases_clean <- df_drug_cases_clean |> 
  dplyr::distinct(processo, disponibilizacao, .keep_all = TRUE) |> 
  dplyr::group_by(processo) |> 
  dplyr::filter(disponibilizacao == min(disponibilizacao)) |> 
  dplyr::filter(sentence_lenght == max(sentence_lenght)) |>
  dplyr::ungroup() |> 
  dplyr::select(-cd_doc) |> 
  dplyr::distinct()

## save the clean dataset
arrow::write_parquet(df_drug_cases_clean, 
                     "data/clean_tjsp_drug_cases_2025.parquet")

# 
# df_drug_cases_clean |> 
#   dplyr::count(group)

# ###### sample
# set.seed(2025)
# 
# ## estimating the size
# 
# n = nrow(df_drug_cases_clean)
# 
# # error margin
# e = 0.05
# 
# # confidence level
# confidence = 0.95
# 
# # proporcion 
# p = 0.5
# 
# sample_size <- sampler::rsampcalc(
#   N = n,
#   e = e*100,
#   ci = confidence*100,
#   p = p
# )
# 
# print(sample_size)
# 
# # adding some margin
# sample_size <- round(sample_size*1.2)
# 
# ## sampling
# # df_drug_cases_clean |>
# #   sampler::ssamp(
# #     n = sample_size,
# #     strata = group
# # )  |> 
# #   dplyr::count(group)
# 
# ## get 200 cases from each group
# df_drug_cases_clean <- df_drug_cases_clean |> 
#   dplyr::group_by(group) |> 
#   dplyr::slice_sample(n = sample_size) |> 
#   dplyr::ungroup() |> 
#   dplyr::select(-sentence_lenght)
# 
# arrow::write_parquet(df_drug_cases_clean, "data/sample400_tjsp_drug_cases_2025_v2.parquet")
# 
