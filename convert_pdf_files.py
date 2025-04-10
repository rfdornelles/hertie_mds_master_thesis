import os
import re
import glob
import time
import numpy as np
import pandas as pd
from pdf2image import convert_from_path
import easyocr
import logging
from tqdm import tqdm

# Configuração do log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializa o leitor com suporte ao português e GPU (se disponível)
reader = easyocr.Reader(['pt'], gpu=True)

def clean_cnj(x):
    """Remove todos os caracteres que não sejam dígitos."""
    return re.sub(r'[^0-9]', '', x)

def aux_read_pdf(path_pdf, overwrite=False, max_attempts=3):
    """
    Processa um arquivo PDF: realiza OCR e salva o resultado em formato Parquet.
    
    Se o arquivo de saída já existir e overwrite=False, ignora o processamento.
    Em caso de falha, tenta novamente até max_attempts.
    """
    destiny = re.sub(r'\.pdf$', '.parquet', path_pdf)
    
    if os.path.exists(destiny) and not overwrite:
        logger.info(f"Arquivo já existe: {destiny}")
        return None

    attempt = 0
    while attempt < max_attempts:
        try:
            # Converter PDF para imagens (ajuste o dpi conforme necessário)
            pages = convert_from_path(path_pdf, dpi=300)
            
            pdf_text = []
            for page in pages:
                image_np = np.array(page)
                resultado = reader.readtext(image_np, detail=0, paragraph=True)
                pdf_text.extend(resultado)
            
            full_text = "\n".join(pdf_text)
            df = pd.DataFrame({
                'file': [path_pdf],
                'text': [full_text]
            })
            
            df.to_parquet(destiny, index=False)
            logger.info(f"OCR concluído para {path_pdf}. Texto salvo em {destiny}")
            return df
        
        except Exception as e:
            attempt += 1
            logger.error(f"Erro ao processar {path_pdf} na tentativa {attempt}: {e}")
            time.sleep(2)  # pausa antes de nova tentativa

    logger.error(f"Falha ao processar {path_pdf} após {max_attempts} tentativas.")
    return None

def read_pdf_from_folder(folder_path):
    """
    Lê todos os arquivos PDF em uma pasta, processa com OCR e agrega os resultados.
    
    A função "limpa" o identificador (cnj) removendo o prefixo 'data/sentencas/' e mantendo somente dígitos.
    """
    cnj = re.sub(r'^data/sentencas/', '', folder_path).strip()
    cnj = clean_cnj(cnj)
    
    files = glob.glob(os.path.join(folder_path, '*.pdf'))
    df_list = []
    for file in tqdm(files, desc=f"Processando PDFs em {folder_path}"):
        result = aux_read_pdf(file)
        if result is not None:
            df_list.append(result)
    
    if df_list:
        df_all = pd.concat(df_list, ignore_index=True)
        df_all['cnj'] = cnj
        return df_all
    else:
        return pd.DataFrame(columns=['file', 'text', 'cnj'])

def list_internal_folders(base_folder="data/sentencas"):
    """
    Lista todas as pastas internas em 'data/sentencas', excluindo a pasta 'outros'.
    """
    folders = []
    for entry in os.scandir(base_folder):
        if entry.is_dir() and os.path.abspath(entry.path) != os.path.abspath(os.path.join(base_folder, "outros")):
            folders.append(entry.path)
    return folders

if __name__ == '__main__':
    base_folder = "data/sentencas"
    internal_folders = list_internal_folders(base_folder)
    
    all_dfs = []
    for folder in tqdm(internal_folders, desc="Processando pastas internas"):
        df_folder = read_pdf_from_folder(folder)
        if not df_folder.empty:
            all_dfs.append(df_folder)
    
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_output = os.path.join(base_folder, "combined_output.parquet")
        final_df.to_parquet(final_output, index=False)
        logger.info(f"Processamento concluído. Dados combinados salvos em {final_output}")
    else:
        logger.info("Nenhum arquivo PDF foi processado.")
