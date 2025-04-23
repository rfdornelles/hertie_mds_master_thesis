# **Information Extraction from Criminal Court Sentences**

## **Objective**
You must extract information from Brazilian criminal court sentences, with an emphasis on drug trafficking and related offenses, filling out all fields in a structured *dataset*.  
- **One JSON object per defendant**: If the sentence includes more than one defendant, generate a separate JSON object for each, keeping the same case number.  
- **Missing data**: If any information is not present or not applicable, use **None**.  
- **Data formats**:
  - Numerical fields (e.g., grams, months) should be **0** if nonexistent.
  - Boolean fields must be **True** or **False**.
  - Text fields should be extracted without omissions (or **None** if not mentioned).  

## **Context**
You will receive the full text of a court ruling (or a hearing transcript containing the ruling). The document generally includes:

- **Header**: “Vistos” or “SENTENÇA”  
- **Report**: narrative of the facts  
- **Legal reasoning**: judicial justification  
- **Dispositive part**: conclusion/final decision  
- **Other information**: testimonies, mentions of the complaint, seizures, legal grounds, etc.

These texts may contain information about the defendant, the judge, the court, the penalties applied, as well as various *flags* and assessments of procedural aspects (for example, whether there was formal or informal confession, whether there was interception, etc.).

## **General Instructions**

1. **Read the full text** and identify, for each mentioned defendant, the specific data related to them, including the case number (which will be the same for all defendants in the same sentence).  
2. **If something is not mentioned**, fill it with **None**. For numeric fields with no information, use **0**.  
3. **Do not add any other fields** besides those listed. Respect the expected data types (text, numeric, or boolean).  
4. **Flags and evaluations** must be interpreted contextually: even if the exact term is not used, identify synonyms or phrases that indicate the presence of that concept (e.g., “confessed” without explicitly stating “confession”).  
5. **The output** should be a JSON array. Each object represents a case/defendant pair with all the fields below.

---

### Column Fields

1. **processo [case_number]**: *(string)*  
   - Case number (e.g., "00316684320178260050").
2. **juiz [judge_name]**: *(string)*  
   - Judge’s name – do not include titles like Dr./Dra.
3. **sexo_juiz [judge_gender]**: *(string)*  
   - Judge’s gender (e.g., "Masculine" or "Feminine").
4. **nome [defendant_name]**: *(string)*  
   - Full name of the defendant.
5. **local [location]**: *(string)*  
   - Location related to the offense, can be: Commerce and Services, Stadium, Alley in Slum, Invasion/Occupation, Residence, Restaurant or similar, Terminal/Station, Public Road, or Unoccupied Area.
6. **maconha [marijuana]**: *(boolean)*  
   - True if marijuana was seized; False otherwise.
7. **maconha_g [marijuana_g]**: *(numeric)*  
   - Quantity of marijuana in grams. 0 if not specified.
8. **maconha_outras [marijuana_derivatives]**: *(boolean)*  
   - True if other forms of marijuana are mentioned (skunk, hashish); False otherwise.
9. **cocaina [cocaine]**: *(boolean)*  
   - True if cocaine was seized; False otherwise.
10. **cocaina_g [cocaine_g]**: *(numeric)*  
   - Quantity of cocaine in grams. 0 if not specified.
11. **crack [crack]**: *(boolean)*  
   - True if crack was seized; False otherwise.
12. **crack_g [crack_g]**: *(numeric)*  
   - Quantity of crack in grams. 0 if not specified.
13. **ecstasy [ecstasy]**: *(boolean)*  
   - True if ecstasy was seized; False otherwise.
14. **ecstasy_g [ecstasy_g]**: *(numeric)*  
   - Quantity of ecstasy in grams. 0 if not specified.
15. **lsd [lsd]**: *(boolean)*  
   - True if LSD was seized; False otherwise.
16. **lsd_g [lsd_g]**: *(numeric)*  
   - Quantity of LSD in grams. 0 if not specified.
17. **sentenca [verdict]**: *(string)*  
   - Final outcome of the ruling: Acquittal, Reclassification, Partially Upheld, or Upheld.
18. **pena_base [base_sentence]**: *(string)*  
   - Base sentence text (e.g., "8y" for 8 years; "9y 6m" for 9 years and 6 months).
19. **tot_pen [total_sentence]**: *(string)*  
   - Total sentence in text (e.g., "10y 2m"), format <YEARS>y <MONTHS>m <DAYS>d.
20. **tot_pen_meses [total_sentence_months]**: *(numeric)*  
   - Total sentence converted to months. 0 if not specified.
21. **outras_drogas [other_drugs]**: *(boolean)*  
   - True if other substances (not marijuana, cocaine, crack, ecstasy, or LSD) are mentioned; False otherwise.
22. **resultado_art_28 [applied_art_28]**: *(boolean)*  
   - True if the ruling applies article 28 (personal use); False if not.
23. **resultado_art_33 [applied_art_33]**: *(boolean)*  
   - True if the ruling applies article 33 (trafficking); False if not.
24. **resultado_art_34 [applied_art_34]**: *(boolean)*  
   - True if the ruling applies article 34; False if not.
25. **resultado_art_35 [applied_art_35]**: *(boolean)*  
   - True if the ruling applies article 35 (association); False if not.
26. **denuncia_art_33 [charged_art_33]**: *(boolean)*  
   - True if the charges cite article 33; False if not.
27. **denuncia_art_34 [charged_art_34]**: *(boolean)*  
   - True if the charges cite article 34; False if not.
28. **denuncia_art_35 [charged_art_35]**: *(boolean)*  
   - True if the charges cite article 35; False if not.
29. **flag_local_de_trafico [flag_trafficking_location]**: *(boolean)*  
   - True if there's indication the location is linked to trafficking (sales point, storage, etc); False if not.
30. **flag_preso_no_momento_da_sentenca [flag_incarcerated_at_ruling]**: *(boolean)*  
   - True if it's indicated that the defendant was or remained imprisoned at the time of the ruling.
31. **flag_confissao_informal [flag_informal_confession]**: *(boolean)*  
   - True if there’s mention of an informal confession; False otherwise.
32. **flag_confissao [flag_formal_confession]**: *(boolean)*  
   - Disregarding informal confessions, True if a formal confession is mentioned; False if not.
33. **flag_denuncia_anonima [flag_anonymous_report]**: *(boolean)*  
   - True if an anonymous report is mentioned; False otherwise.
34. **flag_denuncia [flag_official_report]**: *(boolean)*  
   - After removing anonymous mentions, True if there is a formal report; False otherwise.
35. **flag_atitude_suspeita [flag_suspicious_behavior]**: *(boolean)*  
   - True if there is mention of suspicious behavior by the defendant; False if not.
36. **flag_divergencias_nos_relatos_dos_policiais [flag_police_report_discrepancies]**: *(boolean)*  
   - True if there are inconsistencies in the police officers' accounts; False if not.
37. **flag_investigacao [flag_investigation]**: *(boolean)*  
   - True if the text mentions investigation, inquiry, or operations; False if not.
38. **flag_interceptacao [flag_interception]**: *(boolean)*  
   - True if there is mention of wiretaps or surveillance; False if not.
39. **flag_mandado [flag_warrant]**: *(boolean)*  
   - True if there’s mention of warrant issuance or execution; False if not.
40. **aval_antecedentes [eval_criminal_history]**: *(boolean)*  
   - True if criminal records are assessed; False if not.
41. **aval_conduta [eval_conduct]**: *(boolean)*  
   - True if defendant’s behavior is assessed; False if not.
42. **aval_personalidade [eval_personality]**: *(boolean)*  
   - True if the text evaluates the defendant’s personality; False if not.
43. **aval_natureza [eval_offense_nature]**: *(boolean)*  
   - True if the nature of the offense is assessed; False if not.
44. **aval_quantidade [eval_quantity]**: *(boolean)*  
   - True if the amount of drugs or items seized is assessed; False if not.
45. **aval_variedade [eval_variety]**: *(boolean)*  
   - True if the variety of substances is assessed; False if not.
46. **aval_circunstancias [eval_circumstances]**: *(boolean)*  
   - True if the crime’s circumstances (location, conditions, etc.) are assessed; False if not.
47. **aval_consequencias [eval_consequences]**: *(boolean)*  
   - True if the consequences of the offense (legal, social, personal) are discussed; False if not.
48. **aval_culpabilidade [eval_culpability]**: *(boolean)*  
   - True if culpability or degree of blame is discussed; False if not.
"""

## **Response Format**

- **Return a JSON array**. If there's only one defendant, the array will have just one object.  
- **One object per defendant**, each with all fields listed above.  
- **Example** (single defendant):

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