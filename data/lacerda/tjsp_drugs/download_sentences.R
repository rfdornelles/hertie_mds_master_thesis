## objective: download the sentences from TJ-SP

## dataset: 
dados_sentencas <- nanoparquet::read_parquet("../dados_sentencas_bd.parquet")

## build the list of processos to be downloaded
processos <- dados_sentencas |> 
  dplyr::pull(processo) |> 
  unique()

## download the sentences
dir_sentencas = "sentences"
dir.create(dir_sentencas, showWarnings = FALSE)

# function to safely download it
baixar_sentenca <- function(processo) {
  tryCatch(
    {
      tjsp::tjsp_baixar_cjpg(
        processo = processo,
        diretorio = dir_sentencas
      )
    },
    error = function(e) {
      message(paste("Erro no processo", processo))
    }
  )
}

purrr::walk(
  processos,
  .f = baixar_sentenca,
  .progress = TRUE
)

tbl_sentencas <- tjsp::tjsp_ler_cjpg(diretorio = dir_sentencas)

## download pdfs just in case
dir_pdf <- "sentences/pdf"
dir.create(dir_pdf, showWarnings = FALSE)

tjsp::autenticar()


baixar_pdf <- function(processo) {
  tryCatch(
    {
      tjsp::tjsp_baixar_sentenca_cjpg(
        cd_doc = processo,
        diretorio = dir_pdf
      )
    },
    error = function(e) {
      message(paste("Erro no processo", processo))
    }
  )
}

tbl_sentencas |> 
  dplyr::pull(cd_doc) |> 
  unique() |> 
  purrr::walk(
    .f = baixar_pdf,
    .progress = TRUE
  )

## TODO: gather the pdfs not downloaded
## TODO: read the pdfd of those process not downloaded
## TODO: build the final dataset

