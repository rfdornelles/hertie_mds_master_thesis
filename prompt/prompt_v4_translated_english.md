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
...
48. **aval_culpabilidade [eval_culpability]**: *(boolean)*  
   - True if culpability or degree of blame is discussed; False if not.

---

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