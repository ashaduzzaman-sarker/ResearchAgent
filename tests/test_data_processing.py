"""
Unit tests for data processing module.
"""
import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from src.data_processing import (
    load_config,
    load_and_chunk_pdf,
    expand_df
)


class TestDataProcessing:
    """Test suite for data processing functionality."""

    def test_load_config_success(self, test_config):
        """Test configuration loading."""
        with patch('src.data_processing.open', create=True) as mock_open:
            import yaml
            mock_open.return_value.__enter__.return_value.read.return_value = yaml.dump(test_config)
            config = load_config()
            assert config is not None

    @patch('src.data_processing.PyPDFLoader')
    def test_load_and_chunk_pdf_success(self, mock_loader):
        """Test successful PDF loading and chunking."""
        # Mock PDF document
        mock_doc = Mock()
        mock_doc.page_content = "This is test content for chunking. " * 50
        mock_doc.metadata = {"page": 1}
        
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = [mock_doc]
        mock_loader.return_value = mock_loader_instance
        
        chunks = load_and_chunk_pdf("test.pdf", chunk_size=100, chunk_overlap=10)
        
        assert len(chunks) > 0
        mock_loader.assert_called_once_with("test.pdf")

    def test_load_and_chunk_pdf_missing_file(self):
        """Test PDF loading with missing file."""
        chunks = load_and_chunk_pdf("nonexistent.pdf")
        assert len(chunks) == 0

    @patch('src.data_processing.PyPDFLoader')
    def test_load_and_chunk_pdf_empty_document(self, mock_loader):
        """Test PDF loading with empty document."""
        mock_loader_instance = Mock()
        mock_loader_instance.load.return_value = []
        mock_loader.return_value = mock_loader_instance
        
        chunks = load_and_chunk_pdf("empty.pdf")
        assert len(chunks) == 0

    @patch('src.data_processing.load_and_chunk_pdf')
    def test_expand_df_success(self, mock_chunk_pdf, sample_arxiv_paper):
        """Test successful DataFrame expansion with chunks."""
        # Mock chunks
        mock_chunk1 = Mock()
        mock_chunk1.page_content = "First chunk content"
        mock_chunk2 = Mock()
        mock_chunk2.page_content = "Second chunk content"
        
        mock_chunk_pdf.return_value = [mock_chunk1, mock_chunk2]
        
        df = pd.DataFrame([sample_arxiv_paper])
        expanded_df = expand_df(df, chunk_size=800, chunk_overlap=100)
        
        assert len(expanded_df) == 2
        assert "id" in expanded_df.columns
        assert "chunk" in expanded_df.columns
        assert expanded_df.iloc[0]["id"] == "1706.03762#0"
        assert expanded_df.iloc[1]["id"] == "1706.03762#1"

    def test_expand_df_missing_columns(self):
        """Test DataFrame expansion with missing required columns."""
        df = pd.DataFrame([{"title": "Test"}])
        
        with pytest.raises(ValueError, match="Missing required columns"):
            expand_df(df)

    @patch('src.data_processing.load_and_chunk_pdf')
    def test_expand_df_no_chunks(self, mock_chunk_pdf, sample_arxiv_paper):
        """Test DataFrame expansion when no chunks are generated."""
        mock_chunk_pdf.return_value = []
        
        df = pd.DataFrame([sample_arxiv_paper])
        expanded_df = expand_df(df)
        
        assert len(expanded_df) == 0

    @patch('src.data_processing.load_and_chunk_pdf')
    def test_expand_df_chunk_ids(self, mock_chunk_pdf, sample_arxiv_paper):
        """Test correct generation of prechunk and postchunk IDs."""
        # Mock 3 chunks
        chunks = [Mock(page_content=f"Chunk {i}") for i in range(3)]
        mock_chunk_pdf.return_value = chunks
        
        df = pd.DataFrame([sample_arxiv_paper])
        expanded_df = expand_df(df)
        
        # Check first chunk
        assert expanded_df.iloc[0]["prechunk_id"] == ""
        assert expanded_df.iloc[0]["postchunk_id"] == "1706.03762#1"
        
        # Check middle chunk
        assert expanded_df.iloc[1]["prechunk_id"] == "1706.03762#0"
        assert expanded_df.iloc[1]["postchunk_id"] == "1706.03762#2"
        
        # Check last chunk
        assert expanded_df.iloc[2]["prechunk_id"] == "1706.03762#1"
        assert expanded_df.iloc[2]["postchunk_id"] == ""
