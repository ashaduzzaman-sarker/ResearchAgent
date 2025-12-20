"""
Unit tests for data extraction module.
"""
import pytest
import json
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.data_extraction import (
    load_config,
    load_env,
    extract_arxiv_data,
    download_pdfs
)


class TestDataExtraction:
    """Test suite for data extraction functionality."""

    def test_load_config_success(self, tmp_path, test_config):
        """Test successful configuration loading."""
        config_file = tmp_path / "test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        with patch('src.data_extraction.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = yaml.dump(test_config)
            config = load_config(str(config_file))
            assert config is not None
            assert "data_extraction" in config

    def test_load_config_file_not_found(self):
        """Test configuration loading with missing file."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")

    @patch('src.data_extraction.load_dotenv')
    def test_load_env(self, mock_load_dotenv):
        """Test environment variable loading."""
        load_env()
        mock_load_dotenv.assert_called_once()

    @patch('src.data_extraction.arxiv.Client')
    def test_extract_arxiv_data_success(self, mock_client, tmp_path, sample_arxiv_paper):
        """Test successful ArXiv data extraction."""
        # Mock ArXiv result
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
        
        # Mock client
        mock_client_instance = Mock()
        mock_client_instance.results.return_value = [mock_result]
        mock_client.return_value = mock_client_instance
        
        json_file = tmp_path / "arxiv_dataset.json"
        df = extract_arxiv_data(
            search_query="cat:cs.AI",
            max_results=1,
            json_file_path=str(json_file)
        )
        
        assert len(df) == 1
        assert df.iloc[0]["title"] == sample_arxiv_paper["title"]
        assert json_file.exists()

    @patch('src.data_extraction.requests.get')
    def test_download_pdfs_success(self, mock_get, tmp_path, sample_arxiv_paper):
        """Test successful PDF downloading."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.iter_content = Mock(return_value=[b"PDF content"])
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        df = pd.DataFrame([sample_arxiv_paper])
        df["pdf_file_name"] = None
        
        download_folder = tmp_path / "pdfs"
        download_folder.mkdir(exist_ok=True)
        
        result_df = download_pdfs(df, str(download_folder))
        
        assert result_df["pdf_file_name"].iloc[0] is not None
        assert "1706.03762.pdf" in result_df["pdf_file_name"].iloc[0]

    def test_download_pdfs_empty_dataframe(self):
        """Test PDF downloading with empty DataFrame."""
        df = pd.DataFrame()
        result_df = download_pdfs(df)
        assert result_df.empty

    @patch('src.data_extraction.requests.get')
    def test_download_pdfs_request_failure(self, mock_get, tmp_path, sample_arxiv_paper):
        """Test PDF downloading with request failure."""
        # Mock HTTP error
        mock_get.side_effect = Exception("Connection error")
        
        df = pd.DataFrame([sample_arxiv_paper])
        df["pdf_file_name"] = None
        
        download_folder = tmp_path / "pdfs"
        download_folder.mkdir(exist_ok=True)
        
        result_df = download_pdfs(df, str(download_folder))
        
        # Should have None for failed download
        assert result_df["pdf_file_name"].iloc[0] is None
