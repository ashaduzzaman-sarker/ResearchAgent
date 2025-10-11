import os
import logging
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from serpapi import GoogleSearch
from langchain.tools import tool

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
# RAG Tool
# ================================
@tool
def rag_search(query: str, index_name: str = "research-agent-index", 
               namespace: str = "arxiv_chunks", top_k: int = 5) -> str:
    """
    Retrieve relevant document chunks from Pinecone based on the query.
    
    Args:
        query: The search query string
        index_name: Pinecone index name (default: "research-agent-index")
        namespace: Pinecone namespace (default: "arxiv_chunks")
        top_k: Number of top results to retrieve (default: 5)
    
    Returns:
        str: Formatted string of retrieved documents with metadata
    """
    try:
        # Validate query
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            logger.error("Invalid query provided to rag_search")
            return "Error: Invalid or empty query provided"

        # Initialize embedding model
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.error("OPENAI_API_KEY not found in environment variables")
            return "Error: OPENAI_API_KEY not configured"

        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002", 
            openai_api_key=openai_api_key
        )
        logger.info("Initialized embedding model for RAG search")

        # Generate query embedding
        try:
            query_vector = embeddings.embed_query(query)
        except Exception as e:
            logger.error("Failed to generate query embedding: %s", e)
            return f"Error: Failed to generate embedding - {str(e)}"

        # Initialize Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_api_key:
            logger.error("PINECONE_API_KEY not found in environment variables")
            return "Error: PINECONE_API_KEY not configured"

        try:
            pc = Pinecone(api_key=pinecone_api_key)
            index = pc.Index(index_name)
        except Exception as e:
            logger.error("Failed to connect to Pinecone index '%s': %s", index_name, e)
            return f"Error: Failed to connect to Pinecone - {str(e)}"

        # Query Pinecone
        try:
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True
            )
        except Exception as e:
            logger.error("Failed to query Pinecone: %s", e)
            return f"Error: Failed to query vector database - {str(e)}"

        # Check if results exist
        if not results or not results.get("matches"):
            logger.info("No relevant documents found for query: %s", query)
            return "No relevant research papers found in the database for this query."

        # Format results
        formatted_results = []
        for idx, match in enumerate(results["matches"], 1):
            metadata = match.get("metadata", {})
            score = match.get("score", 0.0)
            
            # Extract metadata fields with defaults
            title = metadata.get("title", "N/A")
            arxiv_id = metadata.get("arxiv_id", "N/A")
            url = metadata.get("url", "N/A")
            summary = metadata.get("summary", "N/A")
            text = metadata.get("text", "N/A")
            
            # Truncate long fields
            summary_truncated = summary[:300] + "..." if len(summary) > 300 else summary
            text_truncated = text[:800] + "..." if len(text) > 800 else text
            
            result_block = f"""
Document {idx}:
Title: {title}
ArXiv ID: {arxiv_id}
URL: {url}
Relevance Score: {score:.4f}
Summary: {summary_truncated}
Content Excerpt: {text_truncated}
{'='*80}
"""
            formatted_results.append(result_block)

        logger.info("Retrieved %d documents for query: %s", len(results["matches"]), query)
        
        final_output = f"Retrieved {len(results['matches'])} relevant research papers:\n\n" + "\n".join(formatted_results)
        return final_output

    except Exception as e:
        logger.error("Unexpected error in rag_search for query '%s': %s", query, e)
        return f"Error retrieving documents: {str(e)}"

# ================================
# Web Search Tool
# ================================
@tool
def web_search(query: str, max_results: int = 3) -> str:
    """
    Perform a web search using SerpAPI Google Search.
    
    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 3)
    
    Returns:
        str: Formatted string of search results with titles, links, and snippets
    """
    try:
        # Validate query
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            logger.error("Invalid query provided to web_search")
            return "Error: Invalid or empty query provided"

        # Get API key
        serpapi_key = os.getenv("SERPAPI_API_KEY")
        if not serpapi_key:
            logger.error("SERPAPI_API_KEY not found in environment variables")
            return "Error: SERPAPI_API_KEY not configured"

        # Perform search
        try:
            params = {
                "q": query,
                "api_key": serpapi_key,
                "num": max_results,
                "engine": "google"
            }
            search = GoogleSearch(params)
            search_results = search.get_dict()
        except Exception as e:
            logger.error("Failed to perform SerpAPI search: %s", e)
            return f"Error: Failed to perform web search - {str(e)}"

        # Extract organic results
        results = search_results.get("organic_results", [])
        
        if not results:
            logger.info("No web search results found for query: %s", query)
            return "No web search results found for this query."

        # Format results
        formatted_results = []
        for idx, result in enumerate(results[:max_results], 1):
            title = result.get("title", "N/A")
            link = result.get("link", "N/A")
            snippet = result.get("snippet", "N/A")
            
            # Truncate snippet if too long
            snippet_truncated = snippet[:400] + "..." if len(snippet) > 400 else snippet
            
            result_block = f"""
Result {idx}:
Title: {title}
URL: {link}
Snippet: {snippet_truncated}
{'-'*80}
"""
            formatted_results.append(result_block)

        logger.info("Retrieved %d web search results for query: %s", len(formatted_results), query)
        
        final_output = f"Retrieved {len(formatted_results)} web search results:\n\n" + "\n".join(formatted_results)
        return final_output

    except Exception as e:
        logger.error("Unexpected error in web_search for query '%s': %s", query, e)
        return f"Error performing web search: {str(e)}"