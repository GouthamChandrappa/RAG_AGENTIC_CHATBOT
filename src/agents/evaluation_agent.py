from typing import List, Dict, Any, Optional
import logging

from crewai import Agent, Task

from src.evaluation.evaluator import RAGEvaluator

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationAgent:
    """
    Agent responsible for evaluating the RAG process and identifying improvements.
    """
    
    def __init__(self, evaluator: Optional[RAGEvaluator] = None):
        """
        Initialize the evaluation agent.
        
        Args:
            evaluator: RAGEvaluator instance for metric calculation
        """
        self.evaluator = evaluator or RAGEvaluator()
        
        # Create the CrewAI agent
        self.agent = Agent(
            role="RAG System Evaluator",
            goal="Thoroughly evaluate RAG system performance and identify improvement areas",
            backstory="""You are an expert at evaluating and optimizing RAG systems.
            You analyze retrieval and generation metrics to identify areas for improvement
            and suggest concrete, actionable optimization strategies.""",
            verbose=True,
            allow_delegation=False
        )
    
    def evaluate_interaction(
        self,
        query: str,
        answer: str,
        retrieved_context: List[Dict[str, Any]],
        expected_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single RAG interaction using DeepEval metrics.
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_context: List of context chunks used for generation
            expected_answer: Optional expected answer for ground truth
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating RAG interaction for query: {query}")
        
        # Extract text from retrieved context
        context_texts = [ctx.get("text", "") for ctx in retrieved_context]
        
        # Add test case to evaluator
        self.evaluator.add_test_case(
            query=query,
            answer=answer,
            retrieved_context=context_texts,
            expected_answer=expected_answer
        )
        
        # Create a single test case for immediate evaluation
        test_case = self.evaluator.create_test_case(
            query=query,
            answer=answer,
            retrieved_context=context_texts,
            expected_answer=expected_answer
        )
        
        # Evaluate metrics for this test case
        metrics_result = self.evaluator.evaluate_metrics(test_case)
        
        return {
            "query": query,
            "answer": answer,
            "metrics": metrics_result
        }
    
    def analyze_metrics(self, metrics_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze metrics results and identify improvement areas.
        
        Args:
            metrics_result: Metrics calculation results
            
        Returns:
            Analysis with improvement suggestions
        """
        # Create analysis task
        analysis_task = Task(
            description=f"""
            Analyze the following RAG evaluation metrics and identify areas for improvement:
            
            Query: {metrics_result.get('query', '')}
            Answer: {metrics_result.get('answer', '')}
            Metrics: {metrics_result.get('metrics', {})}
            
            For each metric:
            1. Interpret what the score means for system performance
            2. Identify strengths and weaknesses based on the score
            3. Suggest concrete improvements to address any weaknesses
            
            Provide an overall assessment and prioritized improvement plan.
            """,
            agent=self.agent,
            expected_output="Detailed analysis of metrics with improvement suggestions"
        )
        
        result = self.agent.execute_task(analysis_task)
        
        return {
            "query": metrics_result.get("query", ""),
            "metrics": metrics_result.get("metrics", {}),
            "analysis": result
        }
        
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report for all test cases.
        
        Returns:
            Evaluation report
        """
        # First run all evaluations
        evaluation_results = self.evaluator.evaluate_all()
        
        # Create report task
        report_task = Task(
            description=f"""
            Generate a comprehensive evaluation report based on these RAG evaluation results:
            
            {evaluation_results}
            
            Your report should include:
            1. Executive summary with key metrics
            2. Detailed analysis of each metric's performance
            3. Pattern analysis across test cases
            4. Prioritized recommendations for system improvement
            5. Concrete action items for implementation
            
            Make your report detailed, insightful, and actionable.
            """,
            agent=self.agent,
            expected_output="Comprehensive RAG evaluation report"
        )
        
        report = self.agent.execute_task(report_task)
        
        return {
            "raw_results": evaluation_results,
            "report": report
        }