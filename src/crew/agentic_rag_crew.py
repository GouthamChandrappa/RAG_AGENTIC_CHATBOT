from typing import List, Dict, Any, Optional
import logging

from crewai import Crew, Process, Task
# Remove this problematic import
# from crewai.tasks import Task as CrewTask

from src.agents.retrieval_agent import RetrievalAgent
from src.agents.generation_agent import GenerationAgent
from src.agents.evaluation_agent import EvaluationAgent
from src.chunking.chunker import DocumentChunker
from src.embedding.embedder import Embedder
from src.retrieval.vector_store import VectorStore
from src.evaluation.evaluator import RAGEvaluator

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgenticRAGCrew:
    """
    A crew of agents working together to implement a RAG system with agentic behavior.
    """
    
    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
        use_semantic_chunking: bool = config.SEMANTIC_CHUNKING,
        eval_metrics: List[str] = config.EVAL_METRICS,
        retrieval_agent: Optional[RetrievalAgent] = None,
        generation_agent: Optional[GenerationAgent] = None,
        evaluation_agent: Optional[EvaluationAgent] = None,
        max_iterations: int = config.MAX_ITERATIONS
    ):
        """
        Initialize the AgenticRAGCrew.
        
        Args:
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
            use_semantic_chunking: Whether to use semantic chunking
            retrieval_agent: Agent for retrieving context
            generation_agent: Agent for generating responses
            evaluation_agent: Agent for evaluating outputs
            max_iterations: Maximum number of refinement iterations
        """
        # Initialize components
        self.chunker = DocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_semantic_chunking=use_semantic_chunking
        )
        self.embedder = Embedder(model_name="all-MiniLM-L6-v2")
        self.vector_store = VectorStore(embedding_dim=384)
        
        # Initialize agents
        self.retrieval_agent = retrieval_agent or RetrievalAgent(
            embedder=self.embedder,
            vector_store=self.vector_store
        )
        self.generation_agent = generation_agent or GenerationAgent()
        self.evaluation_agent = evaluation_agent or EvaluationAgent()
        
        self.max_iterations = max_iterations
        self.eval_metrics = eval_metrics
        
        # Create CrewAI crew
        self.crew = Crew(
            agents=[
                self.retrieval_agent.agent,
                self.generation_agent.agent,
                self.evaluation_agent.agent
            ],
            tasks=[],  # Tasks will be added dynamically
            process=Process.sequential,
            verbose=True
        )
        
        # Store interaction history
        self.interactions = []
        
    def index_document(self, document_path: str) -> int:
        """
        Process and index a document for later retrieval.
        
        Args:
            document_path: Path to the document file
            
        Returns:
            Number of chunks indexed
        """
        logger.info(f"Indexing document: {document_path}")
        
        try:
            # Chunk the document
            chunks = self.chunker.chunk_file(document_path)
            
            if not chunks:
                logger.warning(f"No chunks created from {document_path}")
                return 0
                
            # Generate embeddings for chunks
            embedded_chunks = self.embedder.embed_chunks(chunks)
            
            # Add to vector store
            self.vector_store.add_chunks(embedded_chunks)
            
            logger.info(f"Successfully indexed {len(chunks)} chunks from {document_path}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Error indexing document {document_path}: {str(e)}")
            raise
            
    def index_directory(self, directory_path: str, file_extensions: Optional[List[str]] = None) -> int:
        """
        Process and index all documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            file_extensions: List of file extensions to include
            
        Returns:
            Number of chunks indexed
        """
        logger.info(f"Indexing directory: {directory_path}")
        
        try:
            # Chunk all documents in the directory
            chunks = self.chunker.chunk_directory(
                directory_path=directory_path,
                file_extensions=file_extensions
            )
            
            if not chunks:
                logger.warning(f"No chunks created from {directory_path}")
                return 0
                
            # Generate embeddings for chunks
            embedded_chunks = self.embedder.embed_chunks(chunks)
            
            # Add to vector store
            self.vector_store.add_chunks(embedded_chunks)
            
            logger.info(f"Successfully indexed {len(chunks)} chunks from {directory_path}")
            return len(chunks)
        except Exception as e:
            logger.error(f"Error indexing directory {directory_path}: {str(e)}")
            raise
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query using the agentic RAG approach.
        
        Args:
            query: User query
            
        Returns:
            Results including retrieved context, generated response, and evaluation
        """
        logger.info(f"Processing query: {query}")
        
        # 1. Analysis and retrieval
        retrieval_task = Task(
            description=f"""
            Analyze the following query and retrieve the most relevant context:
            
            Query: {query}
            
            Your task is to:
            1. Analyze the query to understand what information is needed
            2. Retrieve the most relevant context from the knowledge base
            3. Explain why the retrieved context is relevant to the query
            """,
            agent=self.retrieval_agent.agent,
            expected_output="Retrieved context with relevance explanation"
        )
        
        # 2. Get actual retrieved documents programmatically
        retrieved_results = self.retrieval_agent.retrieve(query)
        
        # 3. Generation with context
        generation_task = Task(
            description=f"""
            Generate a response to the query based on the retrieved context:
            
            Query: {query}
            
            Retrieved Context:
            {retrieved_results}
            
            Your task is to:
            1. Create a comprehensive and accurate response based only on the context
            2. Ensure all claims are supported by the context
            3. If the context doesn't contain the answer, acknowledge the limitations
            """,
            agent=self.generation_agent.agent,
            expected_output="Generated response based on context",
            context=[retrieval_task]  # Add context from previous task
        )
        
        # 4. Get actual generated response programmatically
        generated_result = self.generation_agent.generate(
            query=query,
            context=retrieved_results
        )
        
        # 5. Evaluation
        evaluation_task = Task(
            description=f"""
            Evaluate the quality of the RAG system output:
            
            Query: {query}
            Retrieved Context: {retrieved_results}
            Generated Response: {generated_result.get('response', '')}
            
            Your task is to:
            1. Assess the relevance of the retrieved context
            2. Evaluate if the response accurately reflects the context
            3. Identify any issues or areas for improvement
            4. Suggest refinements if needed
            """,
            agent=self.evaluation_agent.agent,
            expected_output="Evaluation results with recommendations",
            context=[retrieval_task, generation_task]  # Add context from previous tasks
        )
        
        # Add tasks to the crew
        self.crew.tasks = [retrieval_task, generation_task, evaluation_task]
        
        # Run the crew
        results = self.crew.kickoff()
        
        # Get metrics using DeepEval
        context_texts = [doc.get('text', '') for doc in retrieved_results]
        evaluator = RAGEvaluator()
        evaluator.add_test_case(
            query=query,
            answer=generated_result.get('response', ''),
            retrieved_context=context_texts
        )
        evaluation_metrics = evaluator.evaluate_all()
        
        # Compile complete results
        complete_results = {
            "query": query,
            "retrieved_context": retrieved_results,
            "answer": generated_result.get('response', ''),  # Use "answer" for compatibility
            "generated_response": generated_result.get('response', ''),
            "crew_results": results,
            "evaluation_metrics": evaluation_metrics,
            "processing_time": 0  # Added for compatibility with main.py
        }
        
        # Store the interaction
        self.interactions.append(complete_results)
        
        return complete_results
    def process_query_direct(self, query: str) -> Dict[str, Any]:
        """
        Process a user query directly without using CrewAI orchestration.
        This is useful for debugging or when simpler processing is needed.
        
        Args:
            query: User query
            
        Returns:
            Processing results with generated answer and evaluation
        """
        logger.info(f"Processing query directly: {query}")
        import time
        start_time = time.time()
        
        try:
            # Step 1: Retrieve relevant context
            retrieval_result = self.retrieval_agent.retrieve_with_analysis(query)
            retrieved_docs = retrieval_result.get("retrieved_documents", [])
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Step 2: Generate response
            generation_result = self.generation_agent.generate(
                query=query,
                context=retrieved_docs
            )
            
            # Ensure we get an answer - handle both "answer" and "response" fields
            answer = generation_result.get("answer", "")
            if not answer:
                answer = generation_result.get("response", "")
            
            # Fallback for empty answers
            if not answer or answer.strip() == "":
                if retrieved_docs:
                    answer = "Based on the retrieved documents, I couldn't generate a specific answer. Please try rephrasing your query."
                else:
                    answer = "I couldn't find relevant information to answer your question. Please ensure documents are properly indexed."
            
            logger.info(f"Generated answer: {answer[:100]}...")
            
            # Step 3: Evaluate
            evaluation_result = None
            try:
                if retrieved_docs and answer:
                    evaluation_result = self.evaluation_agent.evaluate_interaction(
                        query=query,
                        answer=answer,
                        retrieved_context=retrieved_docs
                    )
            except Exception as e:
                logger.error(f"Error in evaluation: {str(e)}")
            if evaluation_result:
                logger.info(f"Evaluation metrics: {evaluation_result.get('metrics', {})}")

            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Store the interaction
            interaction = {
                "query": query,
                "retrieval": {
                    "documents": retrieved_docs,
                    "analysis": retrieval_result.get("analysis", "")
                },
                "answer": answer,
                "response": answer,  # For compatibility
                "evaluation": evaluation_result,
                "timestamp": time.time(),
                "processing_time": processing_time
            }
            self.interactions.append(interaction)
            
            return interaction
        except Exception as e:
            import traceback
            logger.error(f"Error processing query directly: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return a result with the error message as answer for display
            return {
                "query": query,
                "error": str(e),
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "response": f"I encountered an error while processing your query: {str(e)}",
                "processing_time": time.time() - start_time
            }
    
    def get_evaluation_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report for all interactions.
        
        Returns:
            Evaluation report
        """
        if not self.interactions:
            logger.warning("No interactions to evaluate")
            return {"message": "No interactions to evaluate"}
            
        return self.evaluation_agent.generate_evaluation_report()
        
    def reset(self):
        """
        Reset the RAG system, clearing all indexed documents and interactions.
        """
        logger.warning("Resetting the RAG system")
        
        # Delete and recreate the vector store collection
        self.vector_store.delete_collection()
        self.vector_store._init_collection()
        
        # Clear interactions
        self.interactions = []