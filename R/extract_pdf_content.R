## list all pdf files

folder <- "data/sentencas"

folder_internal <- fs::dir_info(path = folder) |> 
  dplyr::filter(path != 'data/sentencas/outros') |> 
  dplyr::pull(path)

### function to enter a folder and read the files

path = folder_internal[1]

## aux function to read and return the content
aux_read_pdf <- function(path_pdf, overwrite = FALSE) {
  
  # check if the file already exists
  destiny <- stringr::str_replace(path_pdf, '\\.pdf$', '.parquet'):
  
  if(file.exists(destiny) & !overwrite) {
    message("File already exists: ", destiny)
    return(NULL)
  }
  
  # read pdf
  pdf_text <- pdftools::pdf_ocr_text(pdf = path_pdf,
                                     lang = "por")
  
  # collapse the result
  pdf_text <- stringr::str_collapse(pdf_text, sep = "\n")

  tbl <- tibble::tibble(
    file = path_pdf,
    text = pdf_text
  )
  
  # save as parquet
  nanoparquet::write_parquet(tbl, file = destiny)
  
  # return the text
  return(tbl)
  
}


read_pdf_from_folder < function(path) {
  
  # clean cnj
  cnj <- stringr::str_remove(path, 'data/sentencas/') |> 
    stringr::str_squish() |> 
    abjutils::clean_cnj()
  
  # list all .pdf from the folder
  files <- fs::dir_ls(path = path, regexp = '\\.pdf$') 
  
  # read all pdfs and merge in a tibble
  tbl <- purrr::map_dfr(.x = files,
                         .f = purrr::possibly(aux_read_pdf)) |> 
    dplyr::mutate(cnj = cnj)
  
}
