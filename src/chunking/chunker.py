import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import PyPDF2

# Import chonkie components
from chonkie import TokenChunker, SemanticChunker
from src.utils.helpers import get_file_extension

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunker:
    """
    Class for chunking documents using Chonkie library.
    """
    
    def __init__(
        self, 
        chunk_size: int = 1000, 
        chunk_overlap: int = 200,
        use_semantic_chunking: bool = False
    ):
        """
        Initialize the DocumentChunker.
        
        Args:
            chunk_size: Target size of chunks in tokens
            chunk_overlap: Overlap between chunks in tokens
            use_semantic_chunking: Whether to use semantic chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic_chunking = use_semantic_chunking
        
        # Initialize the appropriate chunker from Chonkie
        if self.use_semantic_chunking:
            logger.info("Using SemanticChunker for document chunking")
            self.chunker = SemanticChunker(
                embedding_model="minishlab/potion-base-8M",
                threshold=0.5,
                chunk_size=self.chunk_size,
                min_sentences=1
            )
        else:
            logger.info("Using TokenChunker for document chunking")
            self.chunker = TokenChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
    
    def _read_file(self, file_path: str) -> str:
        """
        Read a file's content.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File content as string
        """
        ext = get_file_extension(file_path).lower()
        
        if ext == '.pdf':
            return self._read_pdf(file_path)
        else:
            # Handle text files
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read()
    
    def _read_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text
        """
        logger.info(f"Reading PDF file: {file_path}")
        text = ""
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extract text from each page
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
                    
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {str(e)}")
            return ""
    
    def chunk_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Chunk a single file using Chonkie.
        
        Args:
            file_path: Path to the file to chunk
            
        Returns:
            List of chunks with text and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_name = os.path.basename(file_path)
        
        logger.info(f"Chunking file: {file_name}")
        
        try:
            # Read the file content
            text = self._read_file(file_path)
            
            if not text:
                logger.warning(f"No text extracted from {file_name}")
                return []
            
            # Use Chonkie to chunk the text
            chonkie_chunks = self.chunker(text)
            
            # Convert Chonkie chunks to our format
            chunks = []
            for i, chunk in enumerate(chonkie_chunks):
                chunks.append({
                    "text": chunk.text,
                    "metadata": {
                        "source": file_path,
                        "file_name": file_name,
                        "chunk_id": i,
                        "chunk_type": "semantic" if self.use_semantic_chunking else "token",
                        "token_count": chunk.token_count
                    }
                })
            
            logger.info(f"Created {len(chunks)} chunks from {file_name}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking file {file_name}: {str(e)}")
            raise
    
    def chunk_directory(self, directory_path: str, file_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Chunk all files in a directory.
        
        Args:
            directory_path: Path to directory containing files to chunk
            file_extensions: List of file extensions to include (e.g., ['.pdf', '.txt'])
            
        Returns:
            List of all chunks from all files
        """
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"Directory not found: {directory_path}")
            
        all_chunks = []
        file_extensions = file_extensions or ['.pdf', '.txt', '.md', '.json', '.csv']
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in file_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        file_chunks = self.chunk_file(file_path)
                        all_chunks.extend(file_chunks)
                    except Exception as e:
                        logger.warning(f"Skipping file {file} due to error: {str(e)}")
        
        logger.info(f"Created a total of {len(all_chunks)} chunks from directory {directory_path}")
        return all_chunks