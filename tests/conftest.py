"""
Test configuration and fixtures for ResearchAgent tests.
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("SERPER_API_KEY", "test-serper-key")
    monkeypatch.setenv("SERPAPI_API_KEY", "test-serpapi-key")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")

@pytest.fixture
def test_config():
    """Provide test configuration."""
    return {
        "data_extraction": {
            "arxiv": {
                "search_query": "cat:cs.AI",
                "max_results": 5
            },
            "output": {
                "json_file_path": "tests/data/arxiv_dataset.json",
                "pdf_folder": "tests/data/pdfs"
            }
        },
        "data_processing": {
            "chunking": {
                "chunk_size": 800,
                "chunk_overlap": 100
            },
            "output": {
                "expanded_json_path": "tests/data/expanded_dataset.json"
            }
        },
        "embeddings": {
            "model": "text-embedding-ada-002",
            "batch_size": 32,
            "pinecone": {
                "index_name": "test-research-agent-index",
                "namespace": "test_arxiv_chunks",
                "dimension": 1536
            },
            "output": {
                "embedding_metadata_path": "tests/data/embedding_metadata.json"
            }
        },
        "agent": {
            "llm_model": "gpt-4o-mini",
            "max_iterations": 5,
            "report": {
                "target_word_count": 800,
                "sections": ["Summary", "Findings", "References"]
            },
            "hitl": {
                "enabled": False
            },
            "tools": {
                "rag": {
                    "top_k": 3
                },
                "web": {
                    "provider": "serper",
                    "max_results": 2
                }
            },
            "output": {
                "responses_path": "tests/data/agent_responses.json",
                "reports_path": "tests/data/reports.json"
            }
        }
    }

@pytest.fixture
def sample_arxiv_paper():
    """Sample ArXiv paper data for testing."""
    return {
        "title": "Attention Is All You Need",
        "summary": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "arxiv_id": "1706.03762",
        "pdf_link": "https://arxiv.org/pdf/1706.03762.pdf",
        "url": "https://arxiv.org/abs/1706.03762",
        "published": "2017-06-12",
        "updated": "2017-06-12",
        "pdf_file_name": "tests/data/pdfs/1706.03762.pdf"
    }

@pytest.fixture
def sample_chunk():
    """Sample text chunk for testing."""
    return {
        "id": "1706.03762#0",
        "title": "Attention Is All You Need",
        "summary": "The dominant sequence transduction models...",
        "authors": ["Ashish Vaswani", "Noam Shazeer"],
        "arxiv_id": "1706.03762",
        "url": "https://arxiv.org/abs/1706.03762",
        "chunk": "The Transformer model architecture relies entirely on attention mechanisms...",
        "prechunk_id": "",
        "postchunk_id": "1706.03762#1"
    }

@pytest.fixture(autouse=True)
def setup_test_directories(tmp_path):
    """Setup test directories for each test."""
    test_dirs = ['logs', 'tests/data', 'tests/data/pdfs']
    for dir_path in test_dirs:
        (tmp_path / dir_path).mkdir(parents=True, exist_ok=True)
    yield tmp_path
    # Cleanup happens automatically with tmp_path
