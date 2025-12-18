"""
Unit tests for tools module (RAG and web search).
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.tools import rag_search, web_search


class TestRAGSearch:
    """Test suite for RAG search tool."""

    @patch('src.tools.Pinecone')
    @patch('src.tools.OpenAIEmbeddings')
    def test_rag_search_success(self, mock_embeddings, mock_pinecone, mock_env_vars):
        """Test successful RAG search."""
        # Mock embedding generation
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.1] * 1536
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Mock Pinecone response
        mock_index = Mock()
        mock_match = {
            "id": "1706.03762#0",
            "score": 0.95,
            "metadata": {
                "title": "Attention Is All You Need",
                "arxiv_id": "1706.03762",
                "url": "https://arxiv.org/abs/1706.03762",
                "summary": "Test summary",
                "text": "Test text content",
                "prechunk_id": "",
                "postchunk_id": "1706.03762#1"
            }
        }
        mock_index.query.return_value = {"matches": [mock_match]}
        
        mock_pc = Mock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc
        
        result = rag_search.invoke({
            "query": "What is transformer architecture?",
            "index_name": "test-index",
            "namespace": "test-namespace",
            "top_k": 5
        })
        
        assert result is not None
        assert "Attention Is All You Need" in result
        assert "1706.03762" in result

    @patch('src.tools.OpenAIEmbeddings')
    def test_rag_search_invalid_query(self, mock_embeddings, mock_env_vars):
        """Test RAG search with invalid query."""
        result = rag_search.invoke({
            "query": "",
            "index_name": "test-index",
            "namespace": "test-namespace",
            "top_k": 5
        })
        
        assert "Error" in result or "Invalid" in result

    @patch('src.tools.Pinecone')
    @patch('src.tools.OpenAIEmbeddings')
    def test_rag_search_no_results(self, mock_embeddings, mock_pinecone, mock_env_vars):
        """Test RAG search with no results found."""
        # Mock embedding generation
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.1] * 1536
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Mock Pinecone empty response
        mock_index = Mock()
        mock_index.query.return_value = {"matches": []}
        
        mock_pc = Mock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc
        
        result = rag_search.invoke({
            "query": "test query",
            "index_name": "test-index",
            "namespace": "test-namespace",
            "top_k": 5
        })
        
        assert "No relevant" in result

    @patch('src.tools.OpenAIEmbeddings')
    def test_rag_search_missing_api_key(self, mock_embeddings):
        """Test RAG search with missing API key."""
        mock_embeddings.side_effect = Exception("API key not found")
        
        result = rag_search.invoke({
            "query": "test query",
            "index_name": "test-index",
            "namespace": "test-namespace",
            "top_k": 5
        })
        
        assert "Error" in result


class TestWebSearch:
    """Test suite for web search tool."""

    @patch('src.tools.GoogleSearch')
    def test_web_search_success(self, mock_google_search, mock_env_vars):
        """Test successful web search."""
        mock_search_instance = Mock()
        mock_search_instance.get_dict.return_value = {
            "organic_results": [
                {
                    "title": "Test Result 1",
                    "link": "https://example.com/1",
                    "snippet": "This is a test snippet for result 1"
                },
                {
                    "title": "Test Result 2",
                    "link": "https://example.com/2",
                    "snippet": "This is a test snippet for result 2"
                }
            ]
        }
        mock_google_search.return_value = mock_search_instance
        
        result = web_search.invoke({
            "query": "transformer architecture",
            "max_results": 3
        })
        
        assert result is not None
        assert "Test Result 1" in result
        assert "Test Result 2" in result
        assert "example.com" in result

    def test_web_search_invalid_query(self, mock_env_vars):
        """Test web search with invalid query."""
        result = web_search.invoke({
            "query": "",
            "max_results": 3
        })
        
        assert "Error" in result or "Invalid" in result

    @patch('src.tools.GoogleSearch')
    def test_web_search_no_results(self, mock_google_search, mock_env_vars):
        """Test web search with no results."""
        mock_search_instance = Mock()
        mock_search_instance.get_dict.return_value = {"organic_results": []}
        mock_google_search.return_value = mock_search_instance
        
        result = web_search.invoke({
            "query": "test query",
            "max_results": 3
        })
        
        assert "No web search results" in result

    @patch('src.tools.GoogleSearch')
    def test_web_search_api_error(self, mock_google_search, mock_env_vars):
        """Test web search with API error."""
        mock_google_search.side_effect = Exception("API error")
        
        result = web_search.invoke({
            "query": "test query",
            "max_results": 3
        })
        
        assert "Error" in result

    def test_web_search_missing_api_key(self):
        """Test web search with missing API key."""
        result = web_search.invoke({
            "query": "test query",
            "max_results": 3
        })
        
        assert "Error" in result
