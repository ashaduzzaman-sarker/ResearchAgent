import os
import logging
import pandas as pd
from tqdm import tqdm
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import yaml


# ================================
# Logging Configuration
# ================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/research_agent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ================================
# Config Loader
# ================================
def load_config(config_path="config.yaml"):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


# ================================
# PDF Chunking
# ================================
def load_and_chunk_pdf(pdf_file_name, chunk_size=800, chunk_overlap=100):
    if not pdf_file_name or not os.path.exists(pdf_file_name):
        logger.warning(f"PDF not found: {pdf_file_name}")
        return []

    try:
        loader = PyPDFLoader(pdf_file_name)
        documents = loader.load()
        if not documents:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", " ", ""]
        )

        chunks = splitter.split_documents(documents)
        logger.info(f"Split {pdf_file_name} into {len(chunks)} chunks.")
        return chunks
    except Exception as e:
        logger.error(f"Error loading {pdf_file_name}: {e}")
        return []


# ================================
# Expand DataFrame with Chunks
# ================================
def expand_df(df, chunk_size=800, chunk_overlap=100):
    required_columns = {"pdf_file_name", "arxiv_id", "title", "summary", "authors", "pdf_link", "url"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    expanded = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking PDFs"):
        pdf_path = row["pdf_file_name"]
        chunks = load_and_chunk_pdf(pdf_path, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            expanded.append({
                "id": f"{row['arxiv_id']}#{i}",
                "title": row["title"],
                "summary": row["summary"],
                "authors": row["authors"],
                "arxiv_id": row["arxiv_id"],
                "url": row["url"],
                "chunk": chunk.page_content.strip(),
                "prechunk_id": f"{row['arxiv_id']}#{i-1}" if i > 0 else "",
                "postchunk_id": f"{row['arxiv_id']}#{i+1}" if i < len(chunks)-1 else ""
            })

    expanded_df = pd.DataFrame(expanded)
    logger.info(f"Expanded DataFrame created with {len(expanded_df)} chunks.")
    return expanded_df


# ================================
# Main
# ================================
def main():
    try:
        config = load_config()
        json_path = config["data_extraction"]["output"]["json_file_path"]
        df = pd.read_json(json_path)
        logger.info(f"Loaded {len(df)} papers from {json_path}")

        chunk_conf = config["data_processing"]["chunking"]
        expanded_df = expand_df(
            df,
            chunk_size=chunk_conf["chunk_size"],
            chunk_overlap=chunk_conf["chunk_overlap"]
        )

        output_json = "files/expanded_dataset.json"
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        expanded_df.to_json(output_json, orient="records", indent=4)
        logger.info(f"Saved expanded dataset to {output_json}")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise


if __name__ == "__main__":
    main()
