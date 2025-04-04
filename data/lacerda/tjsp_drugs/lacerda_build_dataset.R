dados_sentencas <- readxl::read_excel(
  "data/lacerda/dados_sentencas_limpo_v2 (com resultados).xlsx", 
  sheet = "BD", guess_max = 10000) |> 
  # clean column names
  janitor::clean_names() |> 
  # clean cnj number
  dplyr::mutate(
    processo = abjutils::clean_cnj(processo)
  )


# remove processos with more than 1 defendant
tbl_processos_to_remove <- dados_sentencas |> 
  dplyr::count(processo, sort = TRUE) |> 
  dplyr::filter(n > 1)

dados_sentencas <- dados_sentencas |> 
  dplyr::anti_join(tbl_processos_to_remove, by = "processo")

dados_sentencas |>
  nanoparquet::write_parquet(
    file = "data/lacerda/dados_sentencas_bd.parquet",
    compression = "gzip"
  )
