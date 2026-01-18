"""
Data Extraction Module

This module handles data extraction from ArXiv, including:
- Searching and fetching paper metadata from ArXiv API
- Downloading PDF files from ArXiv
- Saving metadata to JSON format

The module uses the arxiv Python library for API interactions and
requests for PDF downloads with progress tracking via tqdm.
"""

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
    """
    Load configuration from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
            Defaults to 'config.yaml' in the current directory.
    
    Returns:
        dict: Configuration dictionary containing all settings.
    
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there's an error parsing the YAML file.
    
    Example:
        >>> config = load_config('config.yaml')
        >>> arxiv_settings = config['data_extraction']['arxiv']
    """
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
    """
    Load environment variables from a .env file.
    
    Loads API keys and other sensitive configuration from .env file
    into the environment. This includes:
    - OPENAI_API_KEY
    - PINECONE_API_KEY
    - SERPER_API_KEY (optional)
    - SERPAPI_API_KEY (optional)
    - LANGCHAIN_TRACING_V2
    
    Raises:
        Exception: If there's an error loading the .env file.
    
    Example:
        >>> load_env()
        >>> api_key = os.getenv('OPENAI_API_KEY')
    """
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
    """
    Extract research papers from ArXiv and save metadata to a JSON file.
    
    Searches ArXiv using the provided query, fetches paper metadata,
    and saves it to a JSON file. The function also returns a pandas
    DataFrame for immediate use.
    
    Args:
        search_query (str): ArXiv search query. Can use:
            - Category queries: 'cat:cs.AI' for AI papers
            - Keywords: 'transformer attention'
            - Complex queries: 'cat:cs.AI AND transformer'
        max_results (int): Maximum number of papers to fetch.
        json_file_path (str): Path where the JSON file will be saved.
    
    Returns:
        pd.DataFrame: DataFrame containing paper metadata with columns:
            - title: Paper title
            - summary: Paper abstract
            - authors: List of author names
            - arxiv_id: ArXiv paper ID
            - pdf_link: Direct link to PDF
            - url: ArXiv abstract page URL
            - published: Publication date (YYYY-MM-DD)
            - updated: Last update date (YYYY-MM-DD)
    
    Raises:
        Exception: If there's an error fetching data from ArXiv.
    
    Example:
        >>> df = extract_arxiv_data('cat:cs.AI', max_results=5)
        >>> print(df['title'].tolist())
    """
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
    """
    Download PDF files from ArXiv and add file paths to DataFrame.
    
    Downloads PDFs for all papers in the DataFrame, skipping files
    that already exist locally. Adds a 'pdf_file_name' column with
    the local file path for each paper.
    
    Args:
        df (pd.DataFrame): DataFrame containing ArXiv paper metadata
            with 'pdf_link' and 'arxiv_id' columns.
        download_folder (str): Directory where PDFs will be saved.
    
    Returns:
        pd.DataFrame: Input DataFrame with added 'pdf_file_name' column
            containing local file paths (or None for failed downloads).
    
    Raises:
        Exception: If there's a critical error in the download process.
    
    Note:
        - Uses streaming downloads for memory efficiency
        - Implements retry-friendly design (skips existing files)
        - Individual download failures are logged but don't stop the process
    
    Example:
        >>> df = download_pdfs(df, 'files/pdfs')
        >>> successful = df['pdf_file_name'].notna().sum()
        >>> print(f"Downloaded {successful} PDFs")
    """
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