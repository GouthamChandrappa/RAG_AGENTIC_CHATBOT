import pytest
from unittest.mock import MagicMock, patch
import sys

# Add project root to path
sys.path.insert(0, '.')

from src.retrieval.vector_store import VectorStore
from src.agents.retrieval_agent import RetrievalAgent

# Module-level patch for VectorStore tests
@patch('src.retrieval.vector_store.QdrantClient')
class TestVectorStore:
    def test_init_new_collection(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        VectorStore(collection_name="test_collection")
        mock_client.create_collection.assert_called_once()
    
    def test_init_existing_collection(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        # Create a collection object with the right name attribute
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        # Mock the vectors configuration - must be a proper object with proper values
        vectors_config = MagicMock()
        vectors_config.size = 384  # This must match the embedding_dim
        
        params = MagicMock()
        params.vectors = vectors_config
        
        config = MagicMock()
        config.params = params
        
        collection_info = MagicMock()
        collection_info.config = config
        
        mock_client.get_collection.return_value = collection_info
        
        # Create the vector store
        VectorStore(collection_name="test_collection", embedding_dim=384)
        
        # The create_collection method shouldn't be called for existing collections
        assert not mock_client.create_collection.called
    
    def test_init_dimension_mismatch(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        # Setup collection with proper name attribute
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"  # Ensure this matches
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        mock_client.get_collections.return_value = mock_collections
        
        # Send different dimension to trigger mismatch logic
        mock_config = MagicMock()
        mock_config.config.params.vectors.size = 768  # Different from 384
        mock_client.get_collection.return_value = mock_config
        
        # This should trigger delete+create for dimension mismatch
        VectorStore(collection_name="test_collection", embedding_dim=384)
        
        # Assert delete was called
        mock_client.delete_collection.assert_called_once()
        mock_client.create_collection.assert_called_once()
    
    def test_add_chunks(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        mock_collections = MagicMock()
        mock_collections.collections = []
        mock_client.get_collections.return_value = mock_collections
        
        # Setup mock config with matching dimensions
        mock_config = MagicMock()
        mock_config.config.params.vectors.size = 3
        mock_client.get_collection.return_value = mock_config
        
        vector_store = VectorStore(collection_name="test_collection", embedding_dim=3)
        
        # Clear call history
        mock_client.upsert.reset_mock()
        
        chunks = [
            {"text": "Test chunk 1", "embedding": [0.1, 0.2, 0.3], "metadata": {"source": "test.txt"}},
            {"text": "Test chunk 2", "embedding": [0.4, 0.5, 0.6], "metadata": {"source": "test.txt"}}
        ]
        
        vector_store.add_chunks(chunks)
        mock_client.upsert.assert_called_once()
    
    def test_add_chunks_empty(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        vector_store = VectorStore()
        
        # Clear call history
        mock_client.upsert.reset_mock()
        
        vector_store.add_chunks([])
        assert not mock_client.upsert.called
    
    def test_add_chunks_no_embedding(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        vector_store = VectorStore()
        
        with pytest.raises(ValueError, match="Chunks must contain embeddings"):
            vector_store.add_chunks([{"text": "No embedding"}])
    
    def test_add_chunks_dimension_mismatch(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        vector_store = VectorStore(embedding_dim=384)
        
        # Clear call history
        mock_client.upsert.reset_mock()
        
        vector_store.add_chunks([{"text": "Wrong dim", "embedding": [0.1]}])
        assert not mock_client.upsert.called
    
    def test_search(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        # Setup mock config with matching dimensions
        mock_config = MagicMock()
        mock_config.config.params.vectors.size = 3
        mock_client.get_collection.return_value = mock_config
        
        mock_result = [MagicMock(payload={"text": "Result", "source": "test"}, score=0.95)]
        mock_client.search.return_value = mock_result
        
        vector_store = VectorStore(embedding_dim=3)
        
        # Clear call history
        mock_client.search.reset_mock()
        mock_client.search.return_value = mock_result
        
        results = vector_store.search(query_embedding=[0.1, 0.2, 0.3])
        
        mock_client.search.assert_called_once()
        assert len(results) == 1
    
    def test_search_with_filter(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        # Setup mock config
        mock_config = MagicMock()
        mock_config.config.params.vectors.size = 3
        mock_client.get_collection.return_value = mock_config
        
        vector_store = VectorStore(embedding_dim=3)
        
        # Clear call history
        mock_client.search.reset_mock()
        
        filter_conditions = {"must": [{"key": "source", "match": {"value": "test.txt"}}]}
        vector_store.search(query_embedding=[0.1, 0.2, 0.3], filter_conditions=filter_conditions)
        
        mock_client.search.assert_called_once()
    
    def test_search_empty_embedding(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        vector_store = VectorStore()
        
        with pytest.raises(ValueError, match="Query embedding cannot be empty"):
            vector_store.search(query_embedding=[])
    
    def test_search_dimension_mismatch(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        vector_store = VectorStore(embedding_dim=384)
        
        with pytest.raises(ValueError, match="Query embedding dimension mismatch"):
            vector_store.search(query_embedding=[0.1])
    
    def test_delete_collection(self, mock_qdrant_class):
        mock_client = MagicMock()
        mock_qdrant_class.return_value = mock_client
        
        vector_store = VectorStore()
        
        # Clear call history
        mock_client.delete_collection.reset_mock()
        
        vector_store.delete_collection()
        mock_client.delete_collection.assert_called_once()

class TestRetrievalAgent:
    def test_init(self):
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
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [
            {"text": "Test chunk", "metadata": {"source": "test.txt"}, "score": 0.95}
        ]
        
        agent = RetrievalAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store,
            top_k=10
        )
        
        results = agent.retrieve("test query")
        
        mock_embedder.embed_text.assert_called_once_with("test query")
        mock_vector_store.search.assert_called_once_with(
            query_embedding=[0.1, 0.2, 0.3],
            limit=10
        )
        assert len(results) == 1
    
    def test_retrieve_error_handling(self):
        mock_embedder = MagicMock()
        mock_embedder.embed_text.side_effect = Exception("Error")
        
        agent = RetrievalAgent(
            embedder=mock_embedder,
            vector_store=MagicMock()
        )
        
        results = agent.retrieve("test query")
        assert len(results) == 0
    
    @patch('src.agents.retrieval_agent.Task')
    def test_analyze_query(self, mock_task):
        mock_task_instance = MagicMock()
        mock_task.return_value = mock_task_instance
        
        mock_agent = MagicMock()
        mock_agent.execute_task.return_value = "Analysis result"
        
        agent = RetrievalAgent(
            embedder=MagicMock(),
            vector_store=MagicMock()
        )
        agent.agent = mock_agent
        
        result = agent._analyze_query("test query")
        
        assert mock_task.called
        assert mock_agent.execute_task.called
        assert result["analysis"] == "Analysis result"
    
    def test_retrieve_with_analysis(self):
        mock_embedder = MagicMock()
        mock_vector_store = MagicMock()
        mock_vector_store.search.return_value = [{"text": "Test result"}]
        
        agent = RetrievalAgent(
            embedder=mock_embedder,
            vector_store=mock_vector_store
        )
        
        agent._analyze_query = MagicMock()
        agent._analyze_query.return_value = {"query": "test query", "analysis": "Analysis result"}
        
        result = agent.retrieve_with_analysis("test query")
        
        assert agent._analyze_query.called
        assert mock_vector_store.search.called
        assert "query" in result
        assert "analysis" in result
        assert "retrieved_documents" in result
    
    @patch('src.agents.retrieval_agent.Task')
    def test_evaluate_retrieval(self, mock_task):
        mock_task_instance = MagicMock()
        mock_task.return_value = mock_task_instance
        
        mock_agent = MagicMock()
        mock_agent.execute_task.return_value = "Evaluation result"
        
        agent = RetrievalAgent(
            embedder=MagicMock(),
            vector_store=MagicMock()
        )
        agent.agent = mock_agent
        
        retrieved_docs = [{"text": "Test document"}]
        result = agent.evaluate_retrieval("test query", retrieved_docs)
        
        assert mock_task.called
        assert mock_agent.execute_task.called
        assert result["evaluation"] == "Evaluation result"
    
    def test_evaluate_retrieval_empty(self):
        agent = RetrievalAgent(
            embedder=MagicMock(),
            vector_store=MagicMock()
        )
        
        result = agent.evaluate_retrieval("test query", [])
        
        assert result["evaluation"] == "No documents retrieved"
        assert result["score"] == 0