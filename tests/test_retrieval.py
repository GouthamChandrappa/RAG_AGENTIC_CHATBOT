import pytest
from unittest.mock import MagicMock, patch

from src.retrieval.vector_store import VectorStore
from src.agents.retrieval_agent import RetrievalAgent

class TestVectorStore:
    """Test suite for the VectorStore class."""
    
    @patch('qdrant_client.QdrantClient')
    def test_init(self, mock_qdrant):
        """Test the initialization of VectorStore."""
        # Mock the get_collections method
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_qdrant.return_value.get_collections.return_value = mock_collections
        
        vector_store = VectorStore(collection_name="test_collection")
        
        assert vector_store.collection_name == "test_collection"
        assert mock_qdrant.return_value.create_collection.called
    
    @patch('qdrant_client.QdrantClient')
    def test_add_chunks(self, mock_qdrant):
        """Test adding chunks to the vector store."""
        # Mock the get_collections method
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_qdrant.return_value.get_collections.return_value = mock_collections
        
        vector_store = VectorStore(collection_name="test_collection")
        
        # Create test chunks
        chunks = [
            {
                "text": "This is a test chunk.",
                "embedding": [0.1, 0.2, 0.3],
                "metadata": {"source": "test.txt"}
            },
            {
                "text": "This is another test chunk.",
                "embedding": [0.4, 0.5, 0.6],
                "metadata": {"source": "test.txt"}
            }
        ]
        
        vector_store.add_chunks(chunks)
        
        assert mock_qdrant.return_value.upsert.called
    
    @patch('qdrant_client.QdrantClient')
    def test_search(self, mock_qdrant):
        """Test searching in the vector store."""
        # Mock the get_collections method
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_qdrant.return_value.get_collections.return_value = mock_collections
        
        # Mock the search results
        mock_result = [
            MagicMock(
                payload={"text": "This is a test chunk.", "source": "test.txt"},
                score=0.95
            )
        ]
        mock_qdrant.return_value.search.return_value = mock_result
        
        vector_store = VectorStore(collection_name="test_collection")
        
        # Search for a query
        results = vector_store.search(query_embedding=[0.1, 0.2, 0.3], limit=5)
        
        assert mock_qdrant.return_value.search.called
        assert len(results) == 1
        assert "text" in results[0]
        assert "score" in results[0]
        assert "metadata" in results[0]

class TestRetrievalAgent:
    """Test suite for the RetrievalAgent class."""
    
    def test_init(self):
        """Test the initialization of RetrievalAgent."""
        # Mock dependencies
        mock_embedder = MagicMock()
        mock_vector_store = MagicMock()
        
        agent = RetrievalAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            top_k=5
        )
        
        assert agent.embedder == mock_embedder
        assert agent.vector_store == mock_vector_store
        assert agent.top_k == 5
        assert hasattr(agent, 'agent')
    
    def test_retrieve(self):
        """Test the retrieve method."""
        # Mock dependencies
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [
            {
                "text": "This is a test chunk.",
                "metadata": {"source": "test.txt"},
                "score": 0.95
            }
        ]
        
        agent = RetrievalAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store
        )
        
        results = agent.retrieve("test query")
        
        assert mock_embedder.embed_text.called
        assert mock_vector_store.search.called
        assert len(results) == 1