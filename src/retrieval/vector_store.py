from typing import List, Dict, Any, Optional, Union
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector database for storing and retrieving document chunks.
    Currently implements Qdrant, but could be extended for other vector DBs.
    """
    
    def __init__(
        self, 
        collection_name: str = config.COLLECTION_NAME,
        url: str = config.QDRANT_URL,
        embedding_dim: int = 384  # Updated dimension for all-MiniLM-L6-v2
    ):
        """
        Initialize the vector store.
        
        Args:
            collection_name: Name of the collection to use
            url: URL of the Qdrant server
            embedding_dim: Dimension of the embedding vectors
        """
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        self.embedding_dim = embedding_dim
        
        # Initialize collection if it doesn't exist
        self._init_collection()
        
    def _init_collection(self):
        """
        Initialize the vector collection if it doesn't exist.
        """
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        # Only create collection if it doesn't exist
        if self.collection_name not in collection_names:
            logger.info(f"Creating new collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.embedding_dim,
                    distance=qmodels.Distance.COSINE
                )
            )
        else:
            # Check if existing collection has correct dimensions
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            existing_dim = collection_info.config.params.vectors.size
            
            if existing_dim != self.embedding_dim:
                logger.warning(f"Dimension mismatch in collection. Expected {self.embedding_dim}, found {existing_dim}. Recreating collection.")
                self.client.delete_collection(collection_name=self.collection_name)
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.embedding_dim,
                        distance=qmodels.Distance.COSINE
                    )
                )
            else:
                logger.info(f"Using existing collection: {self.collection_name}")
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Add chunks with embeddings to the vector store.
        
        Args:
            chunks: List of chunks with 'text', 'embedding', and 'metadata' fields
        """
        if not chunks:
            return
            
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        points = []
        for chunk in chunks:
            if "embedding" not in chunk:
                raise ValueError("Chunks must contain embeddings")
                
            # Verify embedding dimension matches expected dimension
            embedding = chunk["embedding"]
            if len(embedding) != self.embedding_dim:
                logger.warning(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(embedding)}.")
                continue
                
            # Generate a unique ID for each chunk
            chunk_id = str(uuid.uuid4())
            
            points.append(
                qmodels.PointStruct(
                    id=chunk_id,
                    vector=chunk["embedding"],
                    payload={
                        "text": chunk["text"],
                        **chunk.get("metadata", {})
                    }
                )
            )
        
        if not points:
            logger.warning("No valid points to add to vector store")
            return
            
        # Add points to the collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        logger.info(f"Successfully added {len(points)} chunks to vector store")
    
    def search(
        self, 
        query_embedding: List[float], 
        limit: int = 5, 
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using the query embedding.
        
        Args:
            query_embedding: Embedding vector of the query
            limit: Maximum number of results to return
            filter_conditions: Additional filter conditions
            
        Returns:
            List of matched chunks with similarity scores
        """
        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")
            
        # Verify embedding dimension
        if len(query_embedding) != self.embedding_dim:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.embedding_dim}, got {len(query_embedding)}")
            
        logger.info(f"Searching vector store with limit {limit}")
        
        filter_obj = None
        if filter_conditions:
            filter_obj = qmodels.Filter(**filter_conditions)
            
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit,
            query_filter=filter_obj
        )
        
        results = []
        for scored_point in search_result:
            results.append({
                "text": scored_point.payload.get("text", ""),
                "metadata": {k: v for k, v in scored_point.payload.items() if k != "text"},
                "score": scored_point.score
            })
            
        return results
    
    def delete_collection(self):
        """
        Delete the current collection.
        """
        logger.warning(f"Deleting collection: {self.collection_name}")
        self.client.delete_collection(collection_name=self.collection_name)