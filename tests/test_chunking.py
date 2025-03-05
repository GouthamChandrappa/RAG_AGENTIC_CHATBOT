import os
import sys
import pytest
from pathlib import Path
import tempfile
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="PyPDF2")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="chonkie")
# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.chunking.chunker import DocumentChunker

class TestDocumentChunker:
    """Test suite for the DocumentChunker class."""
    
    @pytest.fixture
    def sample_text_file(self):
        """Create a temporary text file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            tmp.write(b"This is a sample text file.\nIt has multiple lines.\nThis is for testing the chunking functionality.")
            tmp_path = tmp.name
        
        yield tmp_path
        
        # Cleanup
        os.unlink(tmp_path)
    
    def test_init(self):
        """Test the initialization of DocumentChunker."""
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100, use_semantic_chunking=True)
        
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
        assert chunker.use_semantic_chunking is True
        assert hasattr(chunker, 'chunker')  # Changed from 'semantic_chunker'
        
    def test_chunk_file(self, sample_text_file):
        """Test chunking a single file."""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10, use_semantic_chunking=False)
        
        # Test chunking with token-based chunks
        chunks = chunker.chunk_file(sample_text_file)
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        
        for chunk in chunks:
            assert "text" in chunk
            assert "metadata" in chunk
            assert "source" in chunk["metadata"]
            assert "file_name" in chunk["metadata"]
            assert "chunk_id" in chunk["metadata"]
            assert "chunk_type" in chunk["metadata"]
            assert chunk["metadata"]["chunk_type"] == "token"  # Changed from "fixed"
    
    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        chunker = DocumentChunker()
        
        with pytest.raises(FileNotFoundError):
            chunker.chunk_file("nonexistent_file.txt")
    
    def test_chunk_directory(self, sample_text_file):
        """Test chunking a directory."""
        # Use the directory of the sample file
        dir_path = str(Path(sample_text_file).parent)
        
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10, use_semantic_chunking=False)
        
        chunks = chunker.chunk_directory(dir_path, file_extensions=[".txt"])
        
        assert isinstance(chunks, list)
        assert len(chunks) > 0
    
    def test_invalid_directory_path(self):
        """Test handling of invalid directory paths."""
        chunker = DocumentChunker()
        
        with pytest.raises(NotADirectoryError):
            chunker.chunk_directory("nonexistent_directory")