"""
Tools Module - RAG and Web Search

This module implements the core tools used by the research agent:
1. RAG Search: Semantic search over ArXiv papers using Pinecone vector database
2. Web Search: Current information retrieval using Serper or SerpAPI Google Search

Both tools are implemented as LangChain tools and can be used independently
or composed in agent workflows.
"""

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
    Retrieve relevant document chunks from Pinecone based on semantic similarity.
    
    This tool implements Retrieval-Augmented Generation (RAG) by:
    1. Converting the query to an embedding using OpenAI's embedding model
    2. Searching Pinecone vector database for similar document chunks
    3. Returning formatted results with metadata and relevance scores
    
    Args:
        query (str): The search query string (natural language question or keywords)
        index_name (str): Name of the Pinecone index to search.
            Default: "research-agent-index"
        namespace (str): Pinecone namespace containing the vectors.
            Default: "arxiv_chunks"
        top_k (int): Number of most relevant results to return.
            Default: 5
    
    Returns:
        str: Formatted string containing:
            - Document title and ArXiv ID
            - URL to the paper
            - Relevance score (0.0-1.0, higher is more relevant)
            - Summary excerpt
            - Content excerpt from the matching chunk
            
            Returns error message if search fails.
    
    Environment Variables Required:
        - OPENAI_API_KEY: For generating query embeddings
        - PINECONE_API_KEY: For accessing vector database
    
    Example:
        >>> result = rag_search.invoke({
        ...     "query": "What are transformer models?",
        ...     "top_k": 3
        ... })
        >>> print(result)
        Retrieved 3 relevant research papers:
        
        Document 1:
        Title: Attention Is All You Need
        ...
    
    Note:
        - Scores above 0.8 typically indicate high relevance
        - Results are ordered by relevance (highest first)
        - Long texts are truncated for readability
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
def web_search(query: str, max_results: int = 3, provider: str = "serpapi") -> str:
    """
    Perform a web search using Serper or SerpAPI to get current information from Google.
    
    This tool enables the agent to access up-to-date information beyond
    the research papers in the knowledge base. Useful for:
    - Current events and recent developments
    - General knowledge questions
    - Real-time information (news, statistics, etc.)
    - Supplementary context for research queries
    
    Args:
        query (str): The search query (natural language or keywords)
        max_results (int): Maximum number of search results to return.
            Default: 3
        provider (str): "serper" or "serpapi".
    
    Returns:
        str: Formatted string containing:
            - Result title
            - URL/link to the page
            - Snippet/description from the search result
            
            Returns error message if search fails.
    
    Environment Variables Required:
        - SERPER_API_KEY: For accessing Google Search via Serper
        - SERPAPI_API_KEY: For accessing Google Search via SerpAPI
    
    Example:
        >>> result = web_search.invoke({
        ...     "query": "latest developments in GPT-4",
        ...     "max_results": 3
        ... })
        >>> print(result)
        Retrieved 3 web search results:
        
        Result 1:
        Title: GPT-4 Technical Report
        URL: https://...
        Snippet: ...
    
    Note:
        # Results reflect current web content (unlike static research papers)
        # Snippets are truncated for readability
        # Respects provider rate limits
    """
    try:
        # Validate query
        if not query or not isinstance(query, str) or len(query.strip()) == 0:
            logger.error("Invalid query provided to web_search")
            return "Error: Invalid or empty query provided"

        provider_normalized = (provider or "serpapi").lower()

        if provider_normalized == "serper":
            serper_key = os.getenv("SERPER_API_KEY")
            if not serper_key:
                logger.error("SERPER_API_KEY not found in environment variables")
                return "Error: SERPER_API_KEY not configured"

            try:
                import requests
                headers = {
                    "X-API-KEY": serper_key,
                    "Content-Type": "application/json"
                }
                payload = {"q": query, "num": max_results}
                response = requests.post("https://google.serper.dev/search", headers=headers, json=payload, timeout=30)
                response.raise_for_status()
                search_results = response.json()
                results = search_results.get("organic", [])
            except Exception as e:
                logger.error("Failed to perform Serper search: %s", e)
                return f"Error: Failed to perform web search - {str(e)}"
        else:
            serpapi_key = os.getenv("SERPAPI_API_KEY")
            if not serpapi_key:
                logger.error("SERPAPI_API_KEY not found in environment variables")
                return "Error: SERPAPI_API_KEY not configured"

            try:
                params = {
                    "q": query,
                    "api_key": serpapi_key,
                    "num": max_results,
                    "engine": "google"
                }
                search = GoogleSearch(params)
                search_results = search.get_dict()
                results = search_results.get("organic_results", [])
            except Exception as e:
                logger.error("Failed to perform SerpAPI search: %s", e)
                return f"Error: Failed to perform web search - {str(e)}"
        
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