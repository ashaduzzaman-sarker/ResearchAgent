"""
Integration tests for ResearchAgent pipeline.

These tests validate the end-to-end functionality of the system,
including multiple components working together.
"""
import pytest
import os
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path


class TestPipelineIntegration:
    """Integration tests for the full pipeline."""

    @patch('src.data_extraction.arxiv.Client')
    @patch('src.data_extraction.requests.get')
    def test_extraction_to_processing(
        self, mock_requests, mock_arxiv_client, tmp_path, sample_arxiv_paper
    ):
        """Test data extraction followed by processing."""
        from src.data_extraction import extract_arxiv_data, download_pdfs
        from src.data_processing import expand_df
        
        # Mock ArXiv
        mock_result = Mock()
        mock_result.entry_id = "https://arxiv.org/abs/1706.03762"
        mock_result.title = sample_arxiv_paper["title"]
        mock_result.summary = sample_arxiv_paper["summary"]
        mock_result.authors = [Mock(name=name) for name in sample_arxiv_paper["authors"]]
        mock_result.pdf_url = sample_arxiv_paper["pdf_link"]
        mock_result.published = Mock()
        mock_result.published.strftime = Mock(return_value="2017-06-12")
        mock_result.updated = Mock()
        mock_result.updated.strftime = Mock(return_value="2017-06-12")
        
        mock_client_instance = Mock()
        mock_client_instance.results.return_value = [mock_result]
        mock_arxiv_client.return_value = mock_client_instance
        
        # Mock PDF download
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b"PDF content"])
        mock_response.raise_for_status = Mock()
        mock_requests.return_value = mock_response
        
        # Test extraction
        json_file = tmp_path / "arxiv_dataset.json"
        df = extract_arxiv_data(
            search_query="cat:cs.AI",
            max_results=1,
            json_file_path=str(json_file)
        )
        
        assert len(df) == 1
        assert json_file.exists()
        
        # Test PDF download
        pdf_folder = tmp_path / "pdfs"
        df = download_pdfs(df, str(pdf_folder))
        
        assert df["pdf_file_name"].iloc[0] is not None

    @patch('src.tools.OpenAIEmbeddings')
    @patch('src.tools.Pinecone')
    def test_rag_search_integration(
        self, mock_pinecone, mock_embeddings, mock_env_vars
    ):
        """Test RAG search tool integration."""
        from src.tools import rag_search
        
        # Mock embeddings
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.1] * 1536
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Mock Pinecone
        mock_index = Mock()
        mock_match = {
            "id": "test#0",
            "score": 0.95,
            "metadata": {
                "title": "Test Paper",
                "arxiv_id": "test",
                "url": "https://arxiv.org/abs/test",
                "summary": "Test summary",
                "text": "Test content",
                "prechunk_id": "",
                "postchunk_id": ""
            }
        }
        mock_index.query.return_value = {"matches": [mock_match]}
        
        mock_pc = Mock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc
        
        # Test search
        result = rag_search.invoke({
            "query": "test query",
            "index_name": "test-index",
            "namespace": "test-namespace",
            "top_k": 1
        })
        
        assert "Test Paper" in result
        assert "0.95" in result

    @patch('src.tools.GoogleSearch')
    def test_web_search_integration(self, mock_google_search, mock_env_vars):
        """Test web search tool integration."""
        from src.tools import web_search
        
        mock_search_instance = Mock()
        mock_search_instance.get_dict.return_value = {
            "organic_results": [
                {
                    "title": "Test Result",
                    "link": "https://example.com",
                    "snippet": "Test snippet"
                }
            ]
        }
        mock_google_search.return_value = mock_search_instance
        
        result = web_search.invoke({
            "query": "test query",
            "max_results": 1
        })
        
        # Note: This is a test mock URL, not a real user input
        assert "Test Result" in result
        assert "https://example.com" in result  # nosec - test mock data


class TestAgentWorkflow:
    """Integration tests for agent workflow."""

    @patch('src.agent_graph.ChatOpenAI')
    @patch('src.tools.OpenAIEmbeddings')
    @patch('src.tools.Pinecone')
    def test_agent_classification_and_rag(
        self, mock_pinecone, mock_embeddings, mock_llm, mock_env_vars, test_config
    ):
        """Test agent query classification and RAG search flow."""
        from src.agent_graph import build_graph
        
        # Mock LLM for classification
        mock_llm_instance = Mock()
        mock_response = Mock()
        mock_response.content = '{"tools": ["rag_search"]}'
        mock_llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = mock_llm_instance
        
        # Mock embeddings
        mock_embeddings_instance = Mock()
        mock_embeddings_instance.embed_query.return_value = [0.1] * 1536
        mock_embeddings.return_value = mock_embeddings_instance
        
        # Mock Pinecone
        mock_index = Mock()
        mock_index.query.return_value = {"matches": []}
        mock_pc = Mock()
        mock_pc.Index.return_value = mock_index
        mock_pinecone.return_value = mock_pc
        
        # Mock config
        with patch('src.agent_graph.load_config', return_value=test_config):
            graph = build_graph()
            
            initial_state = {
                "query": "What is transformer architecture?",
                "messages": [],
                "retrieved_docs": "",
                "web_results": "",
                "final_answer": "",
                "tools_to_use": [],
                "iteration": 0
            }
            
            # This should not raise an exception
            result = graph.invoke(initial_state)
            
            assert "tools_to_use" in result
            assert isinstance(result["tools_to_use"], list)


class TestConfigurationHandling:
    """Integration tests for configuration handling."""

    def test_config_loading_across_modules(self, test_config, tmp_path):
        """Test that configuration is properly loaded across modules."""
        import yaml
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        # Test loading in different modules
        from src.data_extraction import load_config as load_config_extraction
        from src.agent_graph import load_config as load_config_agent
        
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = yaml.dump(test_config)
            
            config1 = load_config_extraction(str(config_file))
            config2 = load_config_agent(str(config_file))
            
            assert config1["data_extraction"]["arxiv"]["max_results"] == 5
            assert config2["agent"]["llm_model"] == "gpt-4o-mini"

    def test_env_loading(self, mock_env_vars):
        """Test environment variable loading."""
        from src.data_extraction import load_env
        
        # Should not raise exception
        load_env()
        
        # Check that env vars are accessible
        assert os.getenv("OPENAI_API_KEY") == "test-openai-key"
        assert os.getenv("PINECONE_API_KEY") == "test-pinecone-key"
