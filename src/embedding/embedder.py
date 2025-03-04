from typing import List, Dict, Any
import logging
from sentence_transformers import SentenceTransformer
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Embedder:
    """
    Class for generating embeddings using SentenceTransformer directly.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Initialized embedder with model: {model_name}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate an embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        logger.info("Embedding single text")
        embedding = self.model.encode(text).tolist()
        return embedding
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for multiple chunks.
        
        Args:
            chunks: List of chunks with text
            
        Returns:
            Chunks with embeddings added
        """
        if not chunks:
            logger.warning("No chunks to embed")
            return []
            
        logger.info(f"Embedding {len(chunks)} chunks")
        
        # Extract texts for batch embedding
        texts = [chunk["text"] for chunk in chunks]
        
        try:
            # Encode all texts in a batch for efficiency
            embeddings = self.model.encode(texts)
            
            # Add embeddings back to chunks
            for i, embedding in enumerate(embeddings):
                chunks[i]["embedding"] = embedding.tolist()
                
        except Exception as e:
            logger.error(f"Error batch embedding: {str(e)}")
            # Fall back to individual embedding
            for i, chunk in enumerate(chunks):
                try:
                    embedding = self.model.encode(chunk["text"]).tolist()
                    chunks[i]["embedding"] = embedding
                except Exception as e:
                    logger.error(f"Error embedding chunk {i}: {str(e)}")
                    # Use zero vector with correct dimensions
                    chunks[i]["embedding"] = [0.0] * 384  # Dimension for all-MiniLM-L6-v2
        
        logger.info(f"Successfully embedded {len(chunks)} chunks")
        return chunks