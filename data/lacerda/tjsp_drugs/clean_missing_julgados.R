## objective: clean sentences dataset

## dataset: 
dados_sentencas <- nanoparquet::read_parquet(
  "../dados_sentencas.parquet") |> 
  tibble::as_tibble()

dados_sentencas 

## downloaded cases
tbl_tjsp <- tjsp::tjsp_ler_cjpg(diretorio = "sentences")

# testing the reasonable length of the judgment
# tbl_tjsp |> 
#   dplyr::filter(stringr::str_length(julgado) <= 3000) |> 
#   dplyr::pull(julgado)


valid_tbl_tjsp <- tbl_tjsp |> 
  dplyr::filter(!is.na(julgado) & 
                  stringr::str_length(julgado) > 3000)

# identify the needed cases
list_needed_sentences <- dados_sentencas |> 
  dplyr::distinct(processo) |> 
  dplyr::anti_join(dplyr::distinct(valid_tbl_tjsp, processo)) |> 
  unique() |> 
  dplyr::pull(processo)

tbl_needed_sentences <- dados_sentencas |> 
  dplyr::filter(processo %in% list_needed_sentences) |> 
  dplyr::select(processo:vara) |> 
  unique()


tbl_needed_sentences |> 
  dplyr::arrange(vara) |> 
  writexl::write_xlsx("sentences/manual_download/sentences_needed.xlsx")

## read the pdf files
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

## iterate

tbl_new_pdf <- pdf_files |> 
  purrr::map_dfr(read_pdf_tibble, .progress = TRUE)

tbl_new_pdf
