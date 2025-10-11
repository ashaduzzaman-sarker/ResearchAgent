import os
import logging
import pandas as pd
import yaml
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec

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
# Utility Functions
# ================================
def load_config(config_path="config.yaml"):
    """Load configuration from a YAML file."""
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            logger.info("Configuration loaded successfully from %s", config_path)
        return config
    except FileNotFoundError:
        logger.error("Configuration file %s not found", config_path)
        raise
    except yaml.YAMLError as e:
        logger.error("Error parsing configuration file %s: %s", config_path, e)
        raise

def load_env():
    """Load environment variables from a .env file."""
    try:
        load_dotenv()
        logger.info("Environment variables loaded successfully")
    except Exception as e:
        logger.error("Error loading .env file: %s", e)
        raise

# ================================
# Embedding and Indexing
# ================================
def initialize_pinecone(index_name, dimension=1536, cloud="aws", region="us-east-1"):
    """
    Initialize Pinecone client and ensure the index exists.
    
    Args:
        index_name (str): Name of the Pinecone index.
        dimension (int): Dimension of the embeddings (default: 1536 for text-embedding-ada-002).
        cloud (str): Cloud provider for Pinecone (default: aws).
        region (str): Region for Pinecone (default: us-east-1).
    
    Returns:
        pinecone.Index: Pinecone index object.
    """
    try:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")

        pc = Pinecone(api_key=pinecone_api_key)
        
        # Get list of existing indexes
        existing_indexes = [idx.name for idx in pc.list_indexes()]
        
        if index_name not in existing_indexes:
            logger.info("Creating Pinecone index: %s", index_name)
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region)
            )
            logger.info("Pinecone index %s created successfully", index_name)
        else:
            logger.info("Pinecone index %s already exists", index_name)
            
        index = pc.Index(index_name)
        logger.info("Connected to Pinecone index: %s", index_name)
        return index
        
    except Exception as e:
        logger.error("Failed to initialize Pinecone index %s: %s", index_name, e)
        raise

def generate_and_index_embeddings(df, model_name="text-embedding-ada-002", index_name="research-agent-index",
                                 namespace="arxiv_chunks", batch_size=32, dimension=1536):
    """
    Generate embeddings for text chunks and index them in Pinecone.
    
    Args:
        df (pd.DataFrame): DataFrame with chunked data.
        model_name (str): Name of the embedding model.
        index_name (str): Name of the Pinecone index.
        namespace (str): Pinecone namespace for indexing.
        batch_size (int): Number of chunks to process per batch.
        dimension (int): Embedding dimension.
    
    Returns:
        pd.DataFrame: DataFrame with embedding metadata.
    """
    try:
        # Validate DataFrame
        required_columns = {"id", "chunk", "title", "arxiv_id", "url"}
        missing = required_columns - set(df.columns)
        if missing:
            logger.error("Missing required columns in DataFrame: %s", missing)
            raise ValueError(f"Missing required columns: {missing}")
        
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return pd.DataFrame()

        # Initialize embedding model
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")

        embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=openai_api_key)
        logger.info("Initialized embedding model: %s", model_name)

        # Initialize Pinecone index
        index = initialize_pinecone(index_name, dimension=dimension)

        # Prepare batches
        vectors_to_upsert = []
        successful_embeddings = 0
        failed_embeddings = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating embeddings"):
            chunk = row.get("chunk", "")
            
            # Skip invalid chunks
            if not chunk or not isinstance(chunk, str) or len(chunk.strip()) == 0:
                logger.warning("Skipping row %d: Invalid or empty chunk", idx)
                failed_embeddings += 1
                continue

            try:
                # Generate embedding
                vector = embeddings.embed_query(chunk)
                
                # Prepare metadata
                metadata = {
                    "title": str(row.get("title", ""))[:1000],  # Limit to 1000 chars
                    "arxiv_id": str(row.get("arxiv_id", "")),
                    "url": str(row.get("url", "")),
                    "summary": str(row.get("summary", ""))[:2000],  # Limit to 2000 chars
                    "text": chunk[:5000],  # Limit text to 5000 chars for Pinecone
                    "prechunk_id": str(row.get("prechunk_id", "")),
                    "postchunk_id": str(row.get("postchunk_id", ""))
                }
                
                vectors_to_upsert.append({
                    "id": str(row["id"]),
                    "values": vector,
                    "metadata": metadata
                })
                
                successful_embeddings += 1

                # Upsert in batches
                if len(vectors_to_upsert) >= batch_size:
                    try:
                        index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                        logger.info("Upserted batch of %d vectors to Pinecone namespace '%s'", 
                                  len(vectors_to_upsert), namespace)
                        vectors_to_upsert = []
                    except Exception as e:
                        logger.error("Failed to upsert batch to Pinecone: %s", e)
                        failed_embeddings += len(vectors_to_upsert)
                        vectors_to_upsert = []
                        
            except Exception as e:
                logger.error("Error generating embedding for chunk %s: %s", row.get("id", "unknown"), e)
                failed_embeddings += 1
                continue

        # Upsert remaining vectors
        if vectors_to_upsert:
            try:
                index.upsert(vectors=vectors_to_upsert, namespace=namespace)
                logger.info("Upserted final batch of %d vectors to Pinecone namespace '%s'", 
                          len(vectors_to_upsert), namespace)
            except Exception as e:
                logger.error("Failed to upsert final batch to Pinecone: %s", e)
                failed_embeddings += len(vectors_to_upsert)

        logger.info("Embedding generation completed: %d successful, %d failed", 
                   successful_embeddings, failed_embeddings)
        
        # Get index stats
        try:
            stats = index.describe_index_stats()
            logger.info("Pinecone index stats: %s", stats)
        except Exception as e:
            logger.warning("Could not retrieve index stats: %s", e)
            
        return df

    except Exception as e:
        logger.error("Error in generate_and_index_embeddings: %s", e)
        raise

# ================================
# Main
# ================================
def main():
    """Main function to generate embeddings and index them in Pinecone."""
    try:
        # Load environment and config
        load_env()
        config = load_config()

        # Load expanded DataFrame
        json_path = config["data_processing"]["output"]["expanded_json_path"]
        logger.info("Loading expanded dataset from: %s", json_path)
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Expanded dataset not found at {json_path}. "
                                  "Please run data_processing.py first.")
        
        df = pd.read_json(json_path)
        logger.info("Loaded DataFrame with %d chunks from %s", len(df), json_path)

        if df.empty:
            logger.warning("Loaded DataFrame is empty. Nothing to embed.")
            return pd.DataFrame()

        # Get configuration values
        embedding_conf = config.get("embeddings", {})
        pinecone_conf = embedding_conf.get("pinecone", {})
        
        model_name = embedding_conf.get("model", "text-embedding-ada-002")
        batch_size = embedding_conf.get("batch_size", 32)
        index_name = pinecone_conf.get("index_name", "research-agent-index")
        namespace = pinecone_conf.get("namespace", "arxiv_chunks")
        dimension = pinecone_conf.get("dimension", 1536)

        # Generate and index embeddings
        logger.info("Starting embedding generation and indexing...")
        logger.info("Model: %s, Index: %s, Namespace: %s, Batch size: %d", 
                   model_name, index_name, namespace, batch_size)
        
        df = generate_and_index_embeddings(
            df,
            model_name=model_name,
            index_name=index_name,
            namespace=namespace,
            batch_size=batch_size,
            dimension=dimension
        )

        # Save embedding metadata
        output_conf = embedding_conf.get("output", {})
        output_json = output_conf.get("embedding_metadata_path", "files/embedding_metadata.json")
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        
        df.to_json(output_json, orient="records", indent=4)
        logger.info("Saved embedding metadata to %s", output_json)

        logger.info("Embedding and indexing process completed successfully!")
        return df

    except FileNotFoundError as e:
        logger.error("File not found: %s", e)
        raise
    except KeyError as e:
        logger.error("Missing configuration key: %s", e)
        raise
    except Exception as e:
        logger.error("Fatal error in main: %s", e)
        raise

if __name__ == "__main__":
    main()