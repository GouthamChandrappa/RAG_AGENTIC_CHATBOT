from typing import List, Dict, Any, Optional
import logging

from crewai import Agent, Task
from langchain.schema import BaseMessage, SystemMessage, HumanMessage

from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrievalAgent:
    """
    Agent responsible for retrieving relevant context based on user queries.
    """
    
    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        vector_store: Optional[VectorStore] = None,
        top_k: int = 5
    ):
        """
        Initialize the retrieval agent.
        
        Args:
            embedder: Embedder for query embedding
            vector_store: Vector store for document retrieval
            top_k: Number of documents to retrieve
        """
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store or VectorStore()
        self.top_k = top_k
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="Retrieval Specialist",
            goal="Retrieve the most relevant context for user queries",
            backstory="""You are an expert at understanding user queries and retrieving the most 
            relevant information. You analyze questions carefully to find the perfect context.""",
            verbose=True,
            allow_delegation=False
        )
        
    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context based on the query.
        
        Args:
            query: User query
            
        Returns:
            List of retrieved documents with relevance scores
        """
        logger.info(f"Retrieving context for query: {query}")
        
        try:
            # Generate query embedding
            query_embedding = self.embedder.embed_text(query)
            
            # Search the vector store
            results = self.vector_store.search(
                query_embedding=query_embedding,
                limit=self.top_k
            )
            
            logger.info(f"Retrieved {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
            
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to extract key information needs.
        Uses the LLM to identify what information would best answer the query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with query analysis
        """
        # Task for query analysis
        analysis_task = Task(
            description=f"""
            Analyze the following user query and identify:
            1. The core information need
            2. Key concepts that should be present in relevant documents
            3. Any constraints or preferences mentioned
            
            User Query: {query}
            
            Provide your analysis in a structured format.
            """,
            agent=self.agent,
            expected_output="Query analysis with key information needs"
        )
        
        # Use agent.execute_task instead of task.execute
        result = self.agent.execute_task(analysis_task)
        return {"query": query, "analysis": result}
    
    def retrieve_with_analysis(self, query: str) -> Dict[str, Any]:
        """
        Enhanced retrieval that first analyzes the query before retrieval.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with analysis and retrieved documents
        """
        # First analyze the query
        query_analysis = self._analyze_query(query)
        
        # Then retrieve based on the original query
        retrieved_docs = self.retrieve(query)
        
        return {
            "query": query,
            "analysis": query_analysis.get("analysis", ""),
            "retrieved_documents": retrieved_docs
        }
            
    def evaluate_retrieval(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate the quality of retrieved documents for the query.
        
        Args:
            query: User query
            retrieved_docs: List of retrieved documents
            
        Returns:
            Evaluation results
        """
        if not retrieved_docs:
            return {"query": query, "evaluation": "No documents retrieved", "score": 0}
            
        # Create evaluation task
        evaluation_task = Task(
            description=f"""
            Evaluate the relevance of the retrieved documents for the query.
            
            Query: {query}
            
            Retrieved Documents:
            {retrieved_docs}
            
            For each document, rate its relevance on a scale of 1-10 and explain why.
            Then provide an overall assessment of the retrieval quality.
            """,
            agent=self.agent,
            expected_output="Detailed evaluation of retrieved documents"
        )
        
        result = self.agent.execute_task(evaluation_task)
        
        return {
            "query": query,
            "evaluation": result,
            "documents": retrieved_docs
        }