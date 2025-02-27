## objective: clean sentences dataset

## dataset: 
dados_sentencas <- nanoparquet::read_parquet(
  "../dados_sentencas_bd.parquet") |> 
  tibble::as_tibble()

dados_sentencas 

## downloaded cases from tj sp
tbl_tjsp <- tjsp::tjsp_ler_cjpg(diretorio = "sentences")

# testing the reasonable length of the judgment
# tbl_tjsp |> 
#   dplyr::filter(stringr::str_length(julgado) <= 3000) |> 
#   dplyr::pull(julgado)

# remove too short -- probably they're not valid sentences
valid_tbl_tjsp <- tbl_tjsp |> 
  dplyr::filter(!is.na(julgado),
                  stringr::str_length(julgado) > 3000)

# identify the needed cases -- what do I need to download manually
list_needed_sentences <- dados_sentencas |> 
  dplyr::distinct(processo) |> 
  dplyr::anti_join(dplyr::distinct(valid_tbl_tjsp, processo)) |> 
  unique() |> 
  dplyr::pull(processo)

tbl_needed_sentences <- dados_sentencas |> 
  dplyr::filter(processo %in% list_needed_sentences) |> 
  dplyr::select(processo:vara) |> 
  unique()

## saving an Excel file to do the manual job

# tbl_needed_sentences |> 
#   dplyr::arrange(vara) |> 
#   writexl::write_xlsx("sentences/manual_download/sentences_needed.xlsx")

##### read the pdf files
pdf_files <- fs::dir_ls("sentences/manual_download/", regexp = "\\.pdf$")

# function to create a new tibble
read_pdf_tibble <- function(file) {
  
  # id
  cnj_number <- file |> 
    stringr::str_remove("sentences/manual_download/") |> 
    stringr::str_remove("\\.pdf$")
  
  print(paste("Reading file:", cnj_number))
  
  # read the pdf file
  text <- file |> 
    pdftools::pdf_text() |>
    stringr::str_c(collapse = "") |> 
    stringr::str_squish()
  
  print(paste("File read:", cnj_number, " | Length:", stringr::str_length(text)))

  tibble::tibble(
    processo = cnj_number,
    julgado = text
  )  
}

## iterate to read the files

tbl_new_pdf <- pdf_files |> 
  purrr::map_dfr(read_pdf_tibble, .progress = TRUE)

# sanity check
tbl_new_pdf |> 
  dplyr::filter(is.na(julgado) | stringr::str_length(julgado) <  3000)


# clean the julgado column
tbl_new_pdf <- tbl_new_pdf |> 
  dplyr::mutate(
    sentenca = stringr::str_detect(julgado, "(?i)SENTENÇA|TERMO DE AUDIÊNCIA"),
    julgado = stringr::str_remove(julgado, '^fls. [0-9]* TRIBUNAL DE JUSTIÇA DO ESTADO DE SÃO PAULO COMARCA DE SÃO PAULO FORO CENTRAL CRIMINAL BARRA FUNDA '),
    julgado = stringr::str_remove_all(julgado, "Este documento é cópia do original, assinado digitalmente por (.+?), liberado nos autos em [0-9]{1,2}/[0-9]{1,2}/[0-9]{4} às [0-9]{1,2}:[0-9]{1,2}")
)

# sanity check
tbl_new_pdf |> 
  dplyr::filter(!sentenca | is.na(sentenca)) 


empty_sentences <- tbl_tjsp |> 
  dplyr::mutate(
  sentenca = stringr::str_detect(julgado, "(?i)SENTENÇA|TERMO DE AUDIÊNCIA"),
  ) |> 
  # dplyr::count(sentenca)
  dplyr::filter((!sentenca | is.na(sentenca)) & !processo %in% list_needed_sentences) |> 
  dplyr::pull(processo)

### try to download
dir <- "sentences/manual_download/empty_files"
dir.create(dir, showWarnings = FALSE)

#tjsp::autenticar()

tbl_manual_empty <- tbl_tjsp |> 
  dplyr::filter(processo %in% empty_sentences) |> 
  dplyr::select(processo, vara) |> 
  unique()

## saving an Excel file to do the manual job
# 
# tbl_manual_empty |> 
#   dplyr::arrange(vara) |>
#   writexl::write_xlsx("sentences/manual_download/empty_sentences.xlsx")

# read the new files
pdf_files_empty <- fs::dir_ls(dir, regexp = "\\.pdf$")

tbl_new_pdf_empty <- pdf_files_empty |> 
  purrr::map_dfr(read_pdf_tibble, .progress = TRUE) |> 
  dplyr::mutate(
    processo = stringr::str_remove(processo, 'empty_files/'),
    sentenca = stringr::str_detect(julgado, "(?i)SENTENÇA|TERMO DE AUDIÊNCIA"),
    julgado = stringr::str_remove(julgado, '^fls. [0-9]* TRIBUNAL DE JUSTIÇA DO ESTADO DE SÃO PAULO COMARCA DE SÃO PAULO FORO CENTRAL CRIMINAL BARRA FUNDA '),
    julgado = stringr::str_remove_all(julgado, "Este documento é cópia do original, assinado digitalmente por (.+?), liberado nos autos em [0-9]{1,2}/[0-9]{1,2}/[0-9]{4} às [0-9]{1,2}:[0-9]{1,2}")
  )

# sanity check
# tbl_new_pdf_empty
# tbl_new_pdf
# 
# tbl_new_pdf |> 
#   dplyr::bind_rows(tbl_new_pdf_empty) |> 
#   dplyr::pull(processo) |> 
#   unique()

## build new pdf base
tbl_new_pdf_base <- tbl_new_pdf |> 
  dplyr::bind_rows(tbl_new_pdf_empty)

## substitute the julgado from the original dataset if they are present in the manual download

tbl_tjsp <- tbl_tjsp |> 
  dplyr::left_join(tbl_new_pdf_base, by = "processo") |> 
  dplyr::mutate(
    julgado = dplyr::coalesce(julgado.x, julgado.y)
  ) |> 
  # dplyr::select(-starts_with("julgado."), -sentenca, -cd_doc, 
  #               -pagina, -hora_coleta, -duplicado)
  dplyr::select(processo, julgado)

tbl_tjsp |> 
  dplyr::distinct(processo)

## those will need to be substituted
dados_sentencas |> 
  dplyr::select(processo) |>
  dplyr::left_join(tbl_tjsp, by = "processo", relationship = 'many-to-many') |> 
  dplyr::filter(is.na(julgado) | stringr::str_length(julgado) < 3000) |> 
  tibble::view()

## TODO: substitute bad julgados

### join the data with the original dataset
dados_sentencas <- dados_sentencas |> 
  dplyr::left_join(tbl_tjsp, by = "processo") |> 
  dplyr::relocate(julgado, .after = processo)


# dados_sentencas[1,] |> str()
# dados_sentencas[1,'julgado'] |> dplyr::pull()


### export results
dados_sentencas |> 
  nanoparquet::write_parquet("dados_sentencas_bd_clean.parquet")
