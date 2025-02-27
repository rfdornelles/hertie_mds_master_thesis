####### objective: open lacerda's dataset, remove unecessary features and sample

## read
db_lacerda <- nanoparquet::read_parquet(
  "data/lacerda/tjsp_drugs/dados_sentencas_bd_clean.parquet") |> 
  tibble::as_tibble()

tbl_lacerda <- db_lacerda |> 
  dplyr::filter(!is.na(julgado)) |> 
  ## remove features that are not in the sentence
  dplyr::select(
    -processo_cont,
    -nome_cont,
    -cutis,
    -cutis_agrup,
    -cabelo,
    -objetos,
    -dn,
    -dt_disp,
    -idade,
    -sexo,
    -nenhum_58,
    -nenhum_49,
    -na_59,
    -na_50,
    ## pick the best columns,
    -agravantes33,
    -atenuantes33,
    -sentenca_agrup,
    -aumento33,
    -regime_inicial_agrupado,
  ) |> 
  dplyr::rename(
   paragrafo_4o =  x4o,
   paragrafo_4o_agrupado = x4o_agrupado,
   concurso_formal = conc_form,
  )


## transform the column aspectos in different flags

tbl_lacerda <- tbl_lacerda |> 
  dplyr::mutate(
    flag_local_de_trafico = stringr::str_detect(aspectos, stringr::fixed("Local de tráfico")),
    flag_preso_no_momento_da_sentenca = stringr::str_detect(aspectos, stringr::fixed("Preso no momento da sentença")),
    flag_confissao_informal = stringr::str_detect(aspectos, stringr::fixed("Confissão informal")),
    # Para "Confissão": removemos "Confissão informal" para que não haja sobreposição
    flag_confissao = stringr::str_detect(stringr::str_remove_all(aspectos, stringr::fixed("Confissão informal")), stringr::fixed("Confissão")),
    flag_denuncia_anonima = stringr::str_detect(aspectos, stringr::fixed("Denúncia anônima")),
    # Para "Denúncia": removemos "Denúncia anônima" antes de buscar "Denúncia"
    flag_denuncia = stringr::str_detect(stringr::str_remove_all(aspectos, stringr::fixed("Denúncia anônima")), stringr::fixed("Denúncia")),
    flag_atitude_suspeita = stringr::str_detect(aspectos, stringr::fixed("Atitude suspeita")),
    flag_divergencias_nos_relatos_dos_policiais = stringr::str_detect(aspectos, stringr::fixed("Divergências nos relatos dos policiais")),
    flag_investigacao = stringr::str_detect(aspectos, stringr::fixed("Investigação")),
    flag_interceptacao = stringr::str_detect(aspectos, stringr::fixed("Interceptação")),
    flag_mandado = stringr::str_detect(aspectos, stringr::fixed("Mandado")),
    flag_nacionalidade = stringr::str_detect(aspectos, stringr::fixed("Nacionalidade")),
    flag_revista_vexatoria = stringr::str_detect(aspectos, stringr::fixed("Revista Vexatória"))
  ) |> 
  dplyr::select(-aspectos)

## split in train, validation and test
set.seed(55)

split1 <- rsample::initial_split(tbl_lacerda, prop = 0.85)
train_val <- rsample::training(split1)
test <- rsample::testing(split1)

split2 <- rsample::initial_split(train_val, prop = 0.85)
train <- rsample::training(split2)
validation <- rsample::testing(split2)

## save .parquet
nanoparquet::write_parquet(train, "data/train.parquet")
nanoparquet::write_parquet(validation, "data/validation.parquet")
nanoparquet::write_parquet(test, "data/test.parquet")

