from typing import List, Dict, Any, Optional, Union
import logging
import json
import asyncio
import time
from pathlib import Path
import os
from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

import config

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    """
    Evaluator for RAG system using DeepEval to measure retrieval and generation quality.
    """
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize the RAG evaluator with specified metrics.
        """
        self.metrics = metrics or ["answer_relevancy","contextual_precision", "contextual_recall", "contextual_relevancy","faithfulness" ]  # Default to only fastest metric
        self.results = {}
        self.test_cases = []
        
    def create_test_case(
        self, 
        query: str, 
        answer: str, 
        retrieved_context: List[str], 
        expected_answer: Optional[str] = None
    ) -> LLMTestCase:
        """
        Create a test case for evaluation.
        """
        if expected_answer is None:
            expected_answer = answer
            
        if not retrieved_context:
            retrieved_context = ["No context was retrieved for this query."]
            
        return LLMTestCase(
            input=query,
            actual_output=answer,
            expected_output=expected_answer,
            retrieval_context=retrieved_context
        )
    
    def add_test_case(
        self, 
        query: str, 
        answer: str, 
        retrieved_context: List[str], 
        expected_answer: Optional[str] = None
    ):
        """
        Add a test case to the evaluation.
        """
        test_case = self.create_test_case(
            query=query,
            answer=answer,
            retrieved_context=retrieved_context,
            expected_answer=expected_answer
        )
        self.test_cases.append(test_case)
    
    async def evaluate_metric_async(self, metric_name: str, metric, test_case: LLMTestCase) -> Dict[str, float]:
        """
        Evaluate a single metric asynchronously with timeout protection.
        """
        result = {}
        
        try:
            # Create a task with timeout
            metric_task = asyncio.create_task(self._run_metric(metric, test_case))
            score = await asyncio.wait_for(metric_task, timeout=30.0)  # 30-second timeout
            
            result[metric_name] = score
            
            # Get explanation if available
            if hasattr(metric, "explanation") and metric.explanation:
                result[f"{metric_name}_explanation"] = metric.explanation
                
        except asyncio.TimeoutError:
            logger.warning(f"Metric {metric_name} evaluation timed out")
            result[metric_name] = 0.0
        except Exception as e:
            logger.error(f"Error evaluating {metric_name}: {str(e)}")
            result[metric_name] = 0.0
            
        return result
    
    async def _run_metric(self, metric, test_case: LLMTestCase) -> float:
        """
        Run a metric measurement in a way that can be awaited.
        """
        metric.measure(test_case)
        return metric.score
        
    async def evaluate_metrics_async(self, test_case: LLMTestCase) -> Dict[str, float]:
        """
        Evaluate a single test case using all configured metrics asynchronously.
        """
        metrics_map = {
            "contextual_relevancy": ContextualRelevancyMetric(model="gpt-3.5-turbo"),
            "contextual_precision": ContextualPrecisionMetric(model="gpt-3.5-turbo"),
            "contextual_recall": ContextualRecallMetric(model="gpt-3.5-turbo"),
            "answer_relevancy": AnswerRelevancyMetric(model="gpt-3.5-turbo"),
            "faithfulness": FaithfulnessMetric(model="gpt-3.5-turbo")
        }
        
        tasks = []
        for metric_name in self.metrics:
            if metric_name not in metrics_map:
                logger.warning(f"Metric '{metric_name}' not found, skipping")
                continue
                
            metric = metrics_map[metric_name]
            logger.info(f"Starting evaluation of {metric_name}...")
            
            task = self.evaluate_metric_async(metric_name, metric, test_case)
            tasks.append(task)
        
        # Run all metrics in parallel
        metric_results = await asyncio.gather(*tasks)
        
        # Combine all results into a single dictionary
        combined_results = {}
        for result in metric_results:
            combined_results.update(result)
            
        return combined_results
    
    def evaluate_metrics(self, test_case: LLMTestCase) -> Dict[str, float]:
        """
        Synchronous wrapper for evaluate_metrics_async.
        """
        # Run the async function in a new event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If no event loop exists, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(self.evaluate_metrics_async(test_case))
    
    async def evaluate_all_async(self) -> Dict[str, Any]:
        """
        Evaluate all test cases asynchronously and compile results.
        """
        if not self.test_cases:
            logger.warning("No test cases to evaluate")
            return {"message": "No test cases to evaluate"}
            
        logger.info(f"Evaluating {len(self.test_cases)} test cases")
        
        all_tasks = []
        for i, test_case in enumerate(self.test_cases):
            logger.info(f"Starting evaluation of test case {i+1}/{len(self.test_cases)}")
            task = asyncio.create_task(self.evaluate_metrics_async(test_case))
            all_tasks.append((test_case, task))
        
        all_results = []
        for test_case, task in all_tasks:
            try:
                result = await task
                result["query"] = test_case.input
                result["answer"] = test_case.actual_output
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate test case: {str(e)}")
            
        # Calculate average scores
        avg_scores = {}
        for metric in self.metrics:
            scores = [r.get(metric, 0) for r in all_results if metric in r]
            avg_scores[metric] = sum(scores) / len(scores) if scores else 0
            
        self.results = {
            "test_cases": all_results,
            "average_scores": avg_scores,
            "overall_score": sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0
        }
        
        return self.results
    
    def evaluate_all(self) -> Dict[str, Any]:
        """
        Synchronous wrapper for evaluate_all_async.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        return loop.run_until_complete(self.evaluate_all_async())
    
    def save_results(self, output_path: str = "evaluation_results.json"):
        """
        Save evaluation results to a JSON file.
        """
        if not self.results:
            logger.warning("No results to save")
            return
            
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
            
        logger.info(f"Evaluation results saved to {output_file}")
        
    def get_summary(self) -> str:
        """
        Get a human-readable summary of evaluation results.
        """
        if not self.results:
            return "No evaluation results available"
            
        avg_scores = self.results.get("average_scores", {})
        overall_score = self.results.get("overall_score", 0)
        
        summary = [
            "RAG Evaluation Summary:",
            f"Overall Score: {overall_score:.2f}",
            "\nMetric Scores:"
        ]
        
        for metric, score in avg_scores.items():
            summary.append(f"- {metric}: {score:.2f}")
            
        return "\n".join(summary)
