from typing import List, Dict, Any, Optional
import logging
import json

from crewai import Agent, Task
from openai import OpenAI

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationAgent:
    """
    Agent responsible for generating responses based on retrieved context.
    """
    
    def __init__(self, model_name: str = config.LLM_MODEL):
        """
        Initialize the generation agent.
        
        Args:
            model_name: Name of the LLM model to use
        """
        self.model_name = model_name
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="Content Generator",
            goal="Generate accurate, helpful, and contextually relevant responses",
            backstory="""You are an expert at synthesizing information from multiple sources
            and creating coherent, accurate responses that precisely address user queries.""",
            verbose=True,
            allow_delegation=False
        )
    
    def generate(
        self, 
        query: str, 
        context: List[Dict[str, Any]],
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Generate a response based on the query and retrieved context.
        
        Args:
            query: User query
            context: List of retrieved context documents
            max_tokens: Maximum number of tokens in the response
            
        Returns:
            Generated response with metadata
        """
        logger.info(f"Generating response for query: {query}")
        
        # Format context for the prompt
        formatted_context = ""
        for i, doc in enumerate(context):
            formatted_context += f"\nDocument {i+1}:\n{doc.get('text', '')}\n"
        
        # Log the context length for debugging
        logger.info(f"Context length: {len(formatted_context)} characters, {len(context)} documents")
        
        # If no context is provided, return a message indicating no information was found
        if not context or not formatted_context.strip():
            logger.warning("No context available for generation")
            return {
                "query": query,
                "answer": "I couldn't find relevant information to answer your question.",
                "context_used": [],
                "model": self.model_name
            }
            
        # Use the LLM to generate the response
        try:
            # Log which model we're using
            logger.info(f"Using model: {self.model_name}")
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that provides accurate information based only on the given context. If the context doesn't contain the answer, admit that you don't know rather than making up information."},
                    {"role": "user", "content": f"Based on the following context, please answer this question: {query}\n\nContext:\n{formatted_context}"}
                ],
                max_tokens=max_tokens
            )
            
            # Extract the generated text
            answer = response.choices[0].message.content
            
            # Log a preview of the response
            logger.info(f"Generated answer preview: {answer[:100]}...")
            
            return {
                "query": query,
                "answer": answer,
                "context_used": context,
                "model": self.model_name
            }
        except Exception as e:
            import traceback
            logger.error(f"Error generating response: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                "query": query,
                "answer": f"I'm sorry, I couldn't generate a response due to an error: {str(e)}",
                "error": str(e),
                "context_used": context if context else [],
                "model": self.model_name
            }
    
    def analyze_response_quality(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the quality of the generated response.
        
        Args:
            response: Generated response with context
            
        Returns:
            Analysis results
        """
        if "error" in response:
            return {
                "query": response.get("query", ""),
                "quality": "Error in generation",
                "score": 0
            }
            
        # Create analysis task
        analysis_task = Task(
            description=f"""
            Evaluate the quality of this generated response:
            
            Query: {response.get('query', '')}
            Response: {response.get('answer', '')}
            
            Consider:
            1. Accuracy (does it match the context?)
            2. Completeness (does it fully answer the query?)
            3. Clarity (is it well-written and easy to understand?)
            4. Coherence (is it logically structured?)
            
            Rate each aspect on a scale of 1-10 and provide an overall assessment.
            """,
            agent=self.agent,
            expected_output="Detailed quality analysis of the generated response"
        )
        
        result = self.agent.execute_task(analysis_task)
        
        return {
            "query": response.get("query", ""),
            "answer": response.get("answer", ""),
            "quality_analysis": result
        }