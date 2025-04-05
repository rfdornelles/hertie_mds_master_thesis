# **Extração de Informações de Sentenças Judiciais**

## **Objetivo**
Você deverá extrair informações de sentenças judiciais de processos criminais brasileiros, com ênfase em delitos relacionados a tráfico de drogas e correlatos, preenchendo todos os campos de um *dataset* estruturado.  
- **Um objeto JSON por réu**: Se na sentença houver mais de um réu, gere um objeto JSON separado para cada um, mantendo o mesmo número de processo.  
- **Ausência de dados**: Se alguma informação não estiver presente ou não se aplicar, utilize **None**.  
- **Formatos de dados**:
  - Campos numéricos (ex.: gramas, meses) devem ser **0** quando inexistentes.
  - Campos booleanos devem ser **True** ou **False**.
  - Campos textuais devem ser extraídos sem omissões (ou **None** se não houver menção).  

## **Contexto**
Você receberá a íntegra de uma sentença judicial (ou ata de audiência contendo a sentença). O documento geralmente contém:

- **Cabeçalho**: “Vistos” ou “SENTENÇA”  
- **Relatório**: fatos narrados  
- **Fundamentação**: justificativa jurídica  
- **Parte dispositiva**: conclusão/decisão final  
- **Outras informações**: depoimentos, menções à denúncia, apreensões, fundamentações legais etc.

Esses textos podem conter informações sobre o réu, o juiz, a vara, as penas aplicadas, além de diversas *flags* e avaliações de aspectos processuais (por exemplo, se houve confissão formal ou informal, se havia interceptação, etc.).

## **Instruções Gerais**

1. **Leia o texto integralmente** e identifique, para cada réu mencionado, os dados específicos relacionados a ele, incluindo o número do processo (que será o mesmo para todos os réus da mesma sentença).  
2. **Se algo não for citado**, preencha com **None**. Para campos numéricos sem informação, use **0**.  
3. **Não adicione outros campos** além dos listados. Respeite as tipificações (texto, numérico ou booleano).  
4. **As *flags* e avaliações** devem ser analisadas com base no contexto: mesmo que o texto não use o termo exato, identifique sinônimos e expressões que possam indicar a presença do conceito (por exemplo, “confessou” sem a palavra “confissão”).  
5. **A saída** deve ser um array JSON. Cada objeto representa um par processo/réu com todos os campos abaixo.

---

1. **processo**: *(string)*  
  - Número do processo (ex.: "00316684320178260050").
2. **juiz**: *(string)*  
  - Nome do juiz responsável - não inclua títulos como DR/DRA, etc.
3. **sexo_juiz**: *(string)*  
  - Gênero do juiz (por exemplo, "Masculino" ou "Feminino").
4. **nome**: *(string)*  
  - Nome completo do réu.
5. **local**: *(string)*  
  - Local relacionado ao crime/ocorrência, podendo ser: Comércio e serviços, Estádio, Favela - Viela, Invasão/Ocupação, Residência, Restaurante e afins, Terminal/Estação, Via Pública ou Área não ocupada.
6. **maconha**: *(boolean)*  
  - True se houver apreensão de maconha; False caso contrário.
7. **maconha_g**: *(numérico)*  
  - Quantidade de maconha (gramas). 0 se não informado.
8. **maconha_outras**: *(boolean)*  
  - True se há menção a outras formas de maconha (skank, haxixe); False caso contrário.
9. **cocaina**: *(boolean)*  
  - True se houver apreensão de cocaína; False caso contrário.
10. **cocaina_g**: *(numérico)*  
   - Quantidade de cocaína em gramas. 0 se não informado.
11. **crack**: *(boolean)*  
   - True se houver apreensão de crack; False caso contrário.
12. **crack_g**: *(numérico)*  
   - Quantidade de crack em gramas. 0 se não informado.
13. **ecstasy**: *(boolean)*  
   - True se houver apreensão de ecstasy; False caso contrário.
14. **ecstasy_g**: *(numérico)*  
   - Quantidade de ecstasy em gramas. 0 se não informado.
15. **lsd**: *(boolean)*  
   - True se houver apreensão de LSD; False caso contrário.
16. **lsd_g**: *(numérico)*  
   - Quantidade de LSD em gramas. 0 se não informado.
17. **sentenca**: *(string)*  
   - Resultado final da sentença, podendo ser: Absolvição, Desclassificação, Parcialmente Procedente ou Procedente.
18. **pena_base**: *(string)*  
   - Texto da pena-base (ex.: "8a" para 8 anos; "9a 6m" para 9 anos e 6 meses).
19. **tot_pen**: *(string)*  
   - Pena total em texto (ex.: "10a 2m"), no formato <ANOS>a <MESES>m <DIAS>d.
20. **tot_pen_meses**: *(numérico)*  
   - Pena total convertida em meses. 0 se não informado.
21. **outras_drogas**: *(boolean)*  
   - True se houver menção a outras substâncias além de maconha, cocaína, crack, ecstasy e LSD; False caso contrário.
22. **resultado_art_28**: *(boolean)*  
   - True se o resultado/sentença fizer aplicação do art. 28 (uso pessoal); False se não.
23. **resultado_art_33**: *(boolean)*  
   - True se o resultado/sentença fizer aplicação do art. 33 (tráfico); False se não.
24. **resultado_art_34**: *(boolean)*  
   - True se o resultado/sentença fizer aplicação do art. 34; False se não.
25. **resultado_art_35**: *(boolean)*  
   - True se o resultado/sentença fizer aplicação do art. 35 (associação); False se não.
26. **denuncia_art_33**: *(boolean)*  
   - True se a denúncia alegar o art. 33; False se não.
27. **denuncia_art_34**: *(boolean)*  
   - True se a denúncia alegar o art. 34; False se não.
28. **denuncia_art_35**: *(boolean)*  
   - True se a denúncia alegar o art. 35; False se não.
29. **flag_local_de_trafico**: *(boolean)*  
   - True se houver indicação de local associado a tráfico (ponto de venda, depósito, etc); False se não.
30. **flag_preso_no_momento_da_sentenca**: *(boolean)*  
   - True se há indicação de que o réu estava ou permaneceu preso na ocasião da sentença.
31. **flag_confissao_informal**: *(boolean)*  
   - True se houver menção de confissão de modo informal; False caso contrário.
32. **flag_confissao**: *(boolean)*  
   - Desconsiderando confissões informais, True se houver menção a confissão formal; False se não.
33. **flag_denuncia_anonima**: *(boolean)*  
   - True se houver menção a denúncia anônima; False caso contrário.
34. **flag_denuncia**: *(boolean)*  
   - Após remover menções anônimas, True se houver denúncia formal; False caso contrário.
35. **flag_atitude_suspeita**: *(boolean)*  
   - True se houver menção a atitude ou comportamento suspeito do réu; False se não.
36. **flag_divergencias_nos_relatos_dos_policiais**: *(boolean)*  
   - True se houver divergências nos relatos dos policiais; False se não.
37. **flag_investigacao**: *(boolean)*  
   - True se o texto mencionar investigação, inquérito ou diligências; False se não.
38. **flag_interceptacao**: *(boolean)*  
   - True se houver menção a interceptações (telefônicas ou monitoramento); False se não.
39. **flag_mandado**: *(boolean)*  
   - True se houver menção à expedição ou cumprimento de mandado; False se não.
40. **aval_antecedentes**: *(boolean)*  
   - True se houver avaliação de antecedentes criminais; False se não.
41. **aval_conduta**: *(boolean)*  
   - True se houver avaliação da conduta do réu; False se não.
42. **aval_personalidade**: *(boolean)*  
   - True se o texto avaliar a personalidade do réu; False se não.
43. **aval_natureza**: *(boolean)*  
   - True se houver avaliação da natureza do delito; False se não.
44. **aval_quantidade**: *(boolean)*  
   - True se houver avaliação da quantidade de drogas ou itens apreendidos; False se não.
45. **aval_variedade**: *(boolean)*  
   - True se houver avaliação da variedade de substâncias; False se não.
46. **aval_circunstancias**: *(boolean)*  
   - True se o texto avaliar as circunstâncias do crime (local, condições etc.); False se não.
47. **aval_consequencias**: *(boolean)*  
   - True se o texto abordar as consequências do delito (legais, sociais, pessoais); False se não.
48. **aval_culpabilidade**: *(boolean)*  
   - True se houver menção da culpabilidade ou grau de reprovação; False se não.
   
---

## **Formato da Resposta**

- **Retorne um array JSON**. Se houver apenas um réu, o array conterá apenas um objeto.  
- **Um objeto por réu**, cada qual com todos os campos acima.  
- **Exemplo** (um único réu):

```json
[
    {
        "processo": "00316684320178260050",
        "juiz": "JOÃO DA SILVA",
        "sexo_juiz": "Masculino",
        "nome": "FULANO DE TAL",
        "local": "Via Pública",
        "maconha": true,
        "maconha_g": 150.0,
        "maconha_outras": true,
        "cocaina": false,
        "cocaina_g": 0,
        "crack": false,
        "crack_g": 0,
        "ecstasy": false,
        "ecstasy_g": 0,
        "lsd": false,
        "lsd_g": 0,
        "sentenca": "Procedente",
        "pena_base": "8a",
        "tot_pen": "8a 6m",
        "tot_pen_meses": 102,
        "outras_drogas": false,
        "resultado_art_28": false,
        "resultado_art_33": true,
        "resultado_art_34": false,
        "resultado_art_35": false,
        "denuncia_art_33": true,
        "denuncia_art_34": false,
        "denuncia_art_35": false,
        "flag_local_de_trafico": true,
        "flag_preso_no_momento_da_sentenca": true,
        "flag_confissao_informal": false,
        "flag_confissao": true,
        "flag_denuncia_anonima": false,
        "flag_denuncia": true,
        "flag_atitude_suspeita": false,
        "flag_divergencias_nos_relatos_dos_policiais": false,
        "flag_investigacao": true,
        "flag_interceptacao": false,
        "flag_mandado": false,
        "aval_antecedentes": true,
        "aval_conduta": true,
        "aval_personalidade": false,
        "aval_natureza": true,
        "aval_quantidade": true,
        "aval_variedade": false,
        "aval_circunstancias": false,
        "aval_consequencias": false,
        "aval_culpabilidade": false
    }
]
```
