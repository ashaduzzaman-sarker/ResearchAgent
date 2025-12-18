"""
ResearchAgent - AI-powered research assistant with RAG and web search

This package provides tools for building a research agent that combines:
- Retrieval-Augmented Generation (RAG) over ArXiv papers
- Web search for current information
- LLM-powered answer generation

Main Components:
    - data_extraction: Fetch papers from ArXiv
    - data_processing: Process PDFs and create chunks
    - embeddings: Generate and index embeddings in Pinecone
    - tools: RAG search and web search tools
    - agent_graph: LangGraph workflow for agent orchestration
    - main: Pipeline orchestrator

Example:
    >>> from src.main import run_full_pipeline
    >>> results = run_full_pipeline()
    
    >>> from src.agent_graph import build_graph
    >>> graph = build_graph()
    >>> result = graph.invoke({"query": "What are transformers?"})
"""

__version__ = "1.0.0"
__author__ = "Ashaduzzaman Sarker"

# Note: Imports are done explicitly when needed to avoid circular dependencies
# Users should import directly from submodules:
#   from src.data_extraction import extract_arxiv_data
#   from src.agent_graph import build_graph

__all__ = [
    "__version__",
    "__author__",
]
