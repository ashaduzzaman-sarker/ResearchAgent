import os
import json
import logging
import pandas as pd
import arxiv
import requests
from tqdm import tqdm
from dotenv import load_dotenv
import yaml


# =============================
# Logging Configuration
# =============================
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


# =============================
# Utility Functions
# =============================

def load_config(config_path='config.yaml'):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file '{config_path}' not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        raise


def load_env():
    """Load environment variables from a .env file."""
    try:
        load_dotenv()
        logger.info("Environment variables (.env file) loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading .env file: {e}")
        raise


# =============================
# Data Extraction from ArXiv
# =============================

def extract_arxiv_data(search_query='cat:cs.AI', max_results=10, json_file_path='files/arxiv_dataset.json'):
    """Extract research papers from ArXiv and save metadata to a JSON file."""
    try:
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        logger.info(f"Output directory ensured at {os.path.dirname(json_file_path)}.")

        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        results = list(client.results(search))
        logger.info(f"Fetched {len(results)} papers from ArXiv for query: {search_query}.")

        papers = []
        for result in results:
            try:
                arxiv_id = result.entry_id.split('/')[-1]
                pdf_link = result.pdf_url
                paper = {
                    "title": result.title.strip(),
                    "summary": result.summary.strip(),
                    "authors": [a.name for a in result.authors],
                    "arxiv_id": arxiv_id,
                    "pdf_link": pdf_link,
                    "url": f"https://arxiv.org/abs/{arxiv_id}",
                    "published": result.published.strftime("%Y-%m-%d"),
                    "updated": result.updated.strftime("%Y-%m-%d")
                }
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Error processing paper: {e}")
                continue

        if not papers:
            logger.warning("No papers retrieved.")
            return pd.DataFrame()

        df = pd.DataFrame(papers)
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(papers, f, ensure_ascii=False, indent=4)
        logger.info(f"Metadata saved to {json_file_path}.")
        return df

    except Exception as e:
        logger.error(f"Error in extract_arxiv_data: {e}")
        raise


# =============================
# PDF Download Function
# =============================

def download_pdfs(df, download_folder='files/pdfs'):
    """Download PDFs from ArXiv URLs and add 'pdf_file_name' column."""
    try:
        if df.empty:
            logger.warning("DataFrame is empty. No PDFs to download.")
            return df

        os.makedirs(download_folder, exist_ok=True)
        logger.info(f"Download folder ensured at {download_folder}.")

        pdf_files = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading PDFs"):
            pdf_link = row.get("pdf_link")
            arxiv_id = row.get("arxiv_id")

            if not pdf_link:
                pdf_files.append(None)
                continue

            file_path = os.path.join(download_folder, f"{arxiv_id}.pdf")
            if os.path.exists(file_path):
                logger.info(f"PDF already exists for {arxiv_id}, skipping download.")
                pdf_files.append(file_path)
                continue

            try:
                response = requests.get(pdf_link, stream=True, timeout=30)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    for chunk in response.iter_content(8192):
                        f.write(chunk)
                logger.info(f"Downloaded {arxiv_id}")
                pdf_files.append(file_path)
            except Exception as e:
                logger.error(f"Failed to download {arxiv_id}: {e}")
                pdf_files.append(None)

        df["pdf_file_name"] = pdf_files
        logger.info("PDF download process completed.")
        return df

    except Exception as e:
        logger.error(f"Error in download_pdfs: {e}")
        raise


# =============================
# Main Function
# =============================

def main():
    try:
        load_env()
        config = load_config()

        arxiv_conf = config.get("data_extraction", {}).get("arxiv", {})
        output_conf = config.get("data_extraction", {}).get("output", {})

        df = extract_arxiv_data(
            search_query=arxiv_conf.get("search_query", "cat:cs.AI"),
            max_results=arxiv_conf.get("max_results", 10),
            json_file_path=output_conf.get("json_file_path", "files/arxiv_dataset.json")
        )

        df = download_pdfs(df, output_conf.get("pdf_folder", "files/pdfs"))
        
        # CRITICAL FIX: Save the updated DataFrame with pdf_file_name column back to JSON
        json_file_path = output_conf.get("json_file_path", "files/arxiv_dataset.json")
        df.to_json(json_file_path, orient="records", indent=4)
        logger.info(f"Updated metadata with PDF file paths saved to {json_file_path}.")
        
        logger.info("Data extraction and PDF downloading completed successfully.")

    except Exception as e:
        logger.error(f"Fatal error in main(): {e}")
        raise


if __name__ == "__main__":
    main()