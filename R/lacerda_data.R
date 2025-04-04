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
    -x4o,
    -res_outros,
  ) |> 
  dplyr::rename(
   paragrafo_4o_agrupado = x4o_agrupado,
   concurso_formal = conc_form,
  )


## transform the column aspectos in different flags

tbl_lacerda <- tbl_lacerda |> 
  dplyr::mutate(
    maconha = maconha != "0",
    cocaina = cocaina != "0",
    crack = crack != "0" | is.na(crack),
    ecstasy = ecstasy != "0",
    lsd = lsd != "0",
    # maconha outros
    maconha_outras = haxixe == "Sim" | skank == "Sim",
    # outras drogas
    outras_drogas = outras != "0" | anabolizantes == "Sim" | anorexigenos == "Sim" | lanca_perfume == "Sim" | lanca_perfume == "Sim",
    
    # classificação legal da sentença
    resultado_art_28 = stringr::str_detect(res_drogas, 'Art.+28'), 
    resultado_art_33 = stringr::str_detect(res_drogas, 'Art.+33'),
    resultado_art_34 = stringr::str_detect(res_drogas, 'Art.+34'),
    resultado_art_35 = stringr::str_detect(res_drogas, 'Art.+35'),
    
    # classificação legal da denúncia
    denuncia_art_33 = stringr::str_detect(den_drog, 'Art.+33'),
    denuncia_art_34 = stringr::str_detect(den_drog, 'Art.+34'),
    denuncia_art_35 = stringr::str_detect(den_drog, 'Art.+35'),
      
    # flags
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
    # removed --- flag_nacionalidade = stringr::str_detect(aspectos, stringr::fixed("Nacionalidade")),
    # removed --- flag_revista_vexatoria = stringr::str_detect(aspectos, stringr::fixed("Revista Vexatória")),
    ## outros aspectos avaliados
    aval_antecedentes     = stringr::str_detect(aval_neg_pb, stringr::fixed("Antecedentes")),
    aval_conduta          = stringr::str_detect(aval_neg_pb, stringr::fixed("Conduta")),
    aval_personalidade    = stringr::str_detect(aval_neg_pb, stringr::fixed("Personalidade")),
    aval_natureza         = stringr::str_detect(aval_neg_pb, stringr::fixed("Natureza")),
    aval_quantidade       = stringr::str_detect(aval_neg_pb, stringr::fixed("Quantidade")),
    aval_variedade        = stringr::str_detect(aval_neg_pb, stringr::fixed("Variedade")),
    aval_circunstancias   = stringr::str_detect(aval_neg_pb, stringr::fixed("Circunstâncias")),
    aval_consequencias    = stringr::str_detect(aval_neg_pb, stringr::fixed("Consequências")),
    aval_culpabilidade    = stringr::str_detect(aval_neg_pb, stringr::fixed("Culpabilidade")),
    
    ## clean columns
    pena33 = dplyr::if_else(pena33 == "NA", "0", pena33),
    tot_pen = dplyr::if_else(tot_pen == "NA", "0", tot_pen),
     ) |> 
  dplyr::select(-aspectos, -aval_neg_pb,
                ### removing unecessary features
                -anabolizantes,
                -anorexigenos,
                -lanca_perfume,
                -haxixe,
                -skank,
                -tolueno,
                -outras,
                -den_outros,
                ## remove reprocessed columns
                -res_drogas,
                -den_drog,
                -agravantes33_agrup,
                -atenuantes33_agrup,
                -adolescente,
                -arma_de_fogo,
                -interestadual,
                -concurso_formal,
                -estabelecimento,
                -paragrafo_4o_agrupado,
                -aumento33_agrup,
                -pena_drogas,
                -pena_outros,
                -substituicao_da_pena,
                -regime_inicial
                -confissao,
                -menoridade,
                )

# ## transform in boolean
# tbl_lacerda |> 
#   dplyr::mutate(
#     dplyr::across(
#       .cols = c(confissao, menoridade),
#       .fns = \(x) dplyr::if_else(x == 'Sim', 
#                                  true = TRUE, 
#                                  false = FALSE, 
#                                  missing = FALSE)
#     )
#   ) |> 
#   dplyr::count(confissao, menoridade)


### normalize names

# function to upper and remove special chars
normalize_names <- function(x) {
  x |> 
    stringr::str_to_upper() |> 
    stringr::str_squish() |> 
    abjutils::rm_accent()
}

tbl_lacerda <- tbl_lacerda |> 
  dplyr::mutate(
    dplyr::across(
      .cols = c("juiz", "nome"),
      .fns = ~ normalize_names(.x)
    )
  )



## split in train, validation and test
set.seed(55)

split1 <- rsample::initial_split(tbl_lacerda, prop = 0.85)
train_val <- rsample::training(split1)
test <- rsample::testing(split1)

split2 <- rsample::initial_split(train_val, prop = 0.85)
train <- rsample::training(split2)
validation <- rsample::testing(split2)

## save .parquet
nanoparquet::write_parquet(tbl_lacerda, "data/lacerda_clean.parquet")
nanoparquet::write_parquet(train, "data/train.parquet")
nanoparquet::write_parquet(validation, "data/validation.parquet")
nanoparquet::write_parquet(test, "data/test.parquet")

