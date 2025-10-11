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
    """
    Extract research papers from ArXiv and save metadata to a JSON file.

    Args:
        search_query (str): The search query for ArXiv (e.g., "cat:cs.AI").
        max_results (int): Maximum number of results to fetch.
        json_file_path (str): Path to save the JSON file.

    Returns:
        pd.DataFrame: DataFrame containing the extracted metadata.
    """
    try:
        # Validate inputs
        if not isinstance(max_results, int) or max_results <= 0:
            raise ValueError("max_results must be a positive integer.")
        if not json_file_path.endswith('.json'):
            raise ValueError("json_file_path must end with '.json'.")

        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        logger.info(f"Output directory ensured at {os.path.dirname(json_file_path)}.")

        # Initialize ArXiv client
        client = arxiv.Client()
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        # Fetch results
        results = list(client.results(search))
        logger.info(f"Fetched {len(results)} papers from ArXiv for query: {search_query}.")

        papers = []
        for result in results:
            try:
                papers.append({
                    'title': result.title,
                    'summary': result.summary,
                    'authors': [author.name for author in result.authors],
                    'arxiv_id': result.entry_id.split('/')[-1],
                    'pdf_link': result.pdf_url,
                    'published': result.published.strftime('%Y-%m-%d'),
                    'updated': result.updated.strftime('%Y-%m-%d')
                })
            except Exception as e:
                logger.warning(f"Error processing paper {getattr(result, 'entry_id', 'Unknown')}: {e}")
                continue

        if not papers:
            logger.warning("No papers were successfully retrieved from ArXiv.")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(papers)
        logger.info(f"DataFrame created successfully with {len(df)} records.")

        # Save to JSON
        try:
            with open(json_file_path, 'w', encoding='utf-8') as f:
                json.dump(papers, f, ensure_ascii=False, indent=4)
            logger.info(f"Metadata saved successfully to {json_file_path}.")
        except Exception as e:
            logger.error(f"Failed to save metadata to JSON: {e}")
            raise

        return df

    except Exception as e:
        logger.error(f"Error in extract_arxiv_data: {e}")
        raise


# =============================
# PDF Download Function
# =============================

def download_pdfs(df, download_folder='files/pdfs'):
    """
    Download PDFs from ArXiv URL links in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing 'pdf_link' and 'arxiv_id' columns.
        download_folder (str): Folder to save the downloaded PDFs.

    Returns:
        pd.DataFrame: Updated DataFrame with 'pdf_file_name' column.
    """
    try:
        if df.empty:
            logger.warning("DataFrame is empty. No PDFs to download.")
            return df

        if 'pdf_link' not in df.columns or 'arxiv_id' not in df.columns:
            raise ValueError("DataFrame must contain 'pdf_link' and 'arxiv_id' columns.")

        os.makedirs(download_folder, exist_ok=True)
        logger.info(f"Download folder ensured at {download_folder}.")

        pdf_file_names = []
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Downloading PDFs"):
            pdf_link = row.get('pdf_link')
            arxiv_id = row.get('arxiv_id')

            if not pdf_link or not isinstance(pdf_link, str):
                logger.warning(f"Skipping row {index}: Invalid or missing PDF link.")
                pdf_file_names.append(None)
                continue

            file_path = os.path.join(download_folder, f"{arxiv_id}.pdf")

            # Skip if already downloaded
            if os.path.exists(file_path):
                logger.info(f"PDF already exists for {arxiv_id}, skipping download.")
                pdf_file_names.append(file_path)
                continue

            try:
                response = requests.get(pdf_link, stream=True, timeout=30)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                logger.info(f"Downloaded PDF for {arxiv_id} to {file_path}.")
                pdf_file_names.append(file_path)
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download PDF for {arxiv_id} from {pdf_link}: {e}")
                pdf_file_names.append(None)

        df['pdf_file_name'] = pdf_file_names
        logger.info("PDF download process completed.")
        return df

    except Exception as e:
        logger.error(f"Error in download_pdfs: {e}")
        raise


# =============================
# Main Function
# =============================

def main():
    """Main function to execute ArXiv data extraction and PDF downloading."""
    try:
        load_env()
        config = load_config()

        arxiv_config = config.get('data_extraction', {}).get('arxiv', {})
        output_config = config.get('data_extraction', {}).get('output', {})

        # Extract data
        df = extract_arxiv_data(
            search_query=arxiv_config.get('search_query', 'cat:cs.AI'),
            max_results=arxiv_config.get('max_results', 10),
            json_file_path=output_config.get('json_file_path', 'files/arxiv_dataset.json')
        )

        # Download PDFs
        df = download_pdfs(
            df,
            download_folder=output_config.get('pdf_folder', 'files/pdfs')
        )

        logger.info("Data extraction and PDF downloading completed successfully.")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise


# =============================
# Entry Point
# =============================

if __name__ == "__main__":
    main()
