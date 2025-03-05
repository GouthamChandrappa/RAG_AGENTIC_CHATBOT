import pytest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json
import sys
import asyncio
# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.evaluator import RAGEvaluator
from src.agents.evaluation_agent import EvaluationAgent
from deepeval.test_case import LLMTestCase

class TestRAGEvaluator:
    """Test suite for the RAGEvaluator class."""
    
    def test_init(self):
        """Test the initialization of RAGEvaluator."""
        evaluator = RAGEvaluator(metrics=["contextual_relevancy", "faithfulness"])
        
        assert evaluator.metrics == ["contextual_relevancy", "faithfulness"]
        assert evaluator.results == {}
        assert evaluator.test_cases == []
    
    def test_create_test_case(self):
        """Test creating a test case."""
        evaluator = RAGEvaluator()
        
        test_case = evaluator.create_test_case(
            query="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation.",
            retrieved_context=["RAG is a technique that combines retrieval and generation."],
            expected_answer="RAG is Retrieval-Augmented Generation."
        )
        
        assert test_case.input == "What is RAG?"
        assert test_case.actual_output == "RAG stands for Retrieval-Augmented Generation."
        assert test_case.retrieval_context == ["RAG is a technique that combines retrieval and generation."]
        assert test_case.expected_output == "RAG is Retrieval-Augmented Generation."
    
    def test_add_test_case(self):
        """Test adding a test case."""
        evaluator = RAGEvaluator()
        
        evaluator.add_test_case(
            query="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation.",
            retrieved_context=["RAG is a technique that combines retrieval and generation."],
            expected_answer="RAG is Retrieval-Augmented Generation."
        )
        
        assert len(evaluator.test_cases) == 1
        assert evaluator.test_cases[0].input == "What is RAG?"
    @patch('deepeval.metrics.ContextualRelevancyMetric')
    @patch('deepeval.metrics.FaithfulnessMetric')
    def test_evaluate_metrics(self, mock_faithfulness, mock_relevancy):
        """Test evaluating metrics for a test case."""
        # Create mock instances with the score property
        mock_relevancy_instance = MagicMock()
        mock_relevancy_instance.score = 0.8
        mock_relevancy_instance.explanation = "Good relevancy"
        mock_relevancy.return_value = mock_relevancy_instance
        
        mock_faithfulness_instance = MagicMock()
        mock_faithfulness_instance.score = 0.9
        mock_faithfulness_instance.explanation = "Good faithfulness"
        mock_faithfulness.return_value = mock_faithfulness_instance
        
        # Create mock event loop with proper return value
        mock_loop = MagicMock()
        mock_loop.run_until_complete.return_value = {
            "contextual_relevancy": 0.8,
            "faithfulness": 0.9
        }
        
        # Patch get_event_loop to return our mock
        with patch('asyncio.get_event_loop', return_value=mock_loop):
            evaluator = RAGEvaluator(metrics=["contextual_relevancy", "faithfulness"])
            
            test_case = evaluator.create_test_case(
                query="What is RAG?",
                answer="RAG stands for Retrieval-Augmented Generation.",
                retrieved_context=["RAG is a technique that combines retrieval and generation."]
            )
            
            results = evaluator.evaluate_metrics(test_case)
            
            assert "contextual_relevancy" in results
            assert "faithfulness" in results
            assert results["contextual_relevancy"] == 0.8
            assert results["faithfulness"] == 0.9
    
    
    def test_save_results(self):
        """Test saving evaluation results."""
        evaluator = RAGEvaluator()
        
        # Set some dummy results
        evaluator.results = {
            "average_scores": {"contextual_relevancy": 0.8, "faithfulness": 0.9},
            "overall_score": 0.85
        }
        
        # Use a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            evaluator.save_results(tmp_path)
            
            # Check if the file was created and contains the expected content
            assert os.path.exists(tmp_path)
            
            with open(tmp_path, "r") as f:
                saved_data = json.load(f)
                
            assert "average_scores" in saved_data
            assert "overall_score" in saved_data
            assert saved_data["overall_score"] == 0.85
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class TestEvaluationAgent:
    """Test suite for the EvaluationAgent class."""
    
    def test_init(self):
        """Test the initialization of EvaluationAgent."""
        # Mock evaluator
        mock_evaluator = MagicMock()
        
        agent = EvaluationAgent(evaluator=mock_evaluator)
        
        assert agent.evaluator == mock_evaluator
        assert hasattr(agent, 'agent')
    
    def test_evaluate_interaction(self):
        """Test evaluating a RAG interaction."""
        # Create a mock test case
        mock_test_case = MagicMock(spec=LLMTestCase)
        
        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.create_test_case.return_value = mock_test_case
        mock_evaluator.evaluate_metrics.return_value = {
            "contextual_relevancy": 0.8,
            "faithfulness": 0.9
        }
        
        agent = EvaluationAgent(evaluator=mock_evaluator)
        
        # Mock context texts
        context_with_text = [{"text": "RAG is a technique."}]
        
        result = agent.evaluate_interaction(
            query="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation.",
            retrieved_context=context_with_text
        )
        
        # Verify the evaluator methods were called with proper arguments
        mock_evaluator.add_test_case.assert_called_once()
        mock_evaluator.create_test_case.assert_called_once()
        mock_evaluator.evaluate_metrics.assert_called_once_with(mock_test_case)
        
        # Check the expected result structure
        assert "query" in result
        assert "answer" in result
        assert "metrics" in result
        assert result["metrics"] == {"contextual_relevancy": 0.8, "faithfulness": 0.9}

