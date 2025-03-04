import pytest
from unittest.mock import MagicMock, patch
import tempfile
import os
import json

from src.evaluation.evaluator import RAGEvaluator
from src.agents.evaluation_agent import EvaluationAgent

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
        assert test_case.context == ["RAG is a technique that combines retrieval and generation."]
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
        # Mock the metrics
        mock_relevancy.return_value.score = 0.8
        mock_relevancy.return_value.explanation = "Good relevancy"
        
        mock_faithfulness.return_value.score = 0.9
        mock_faithfulness.return_value.explanation = "Good faithfulness"
        
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
    
    @patch('src.evaluation.evaluator.RAGEvaluator.evaluate_metrics')
    def test_evaluate_interaction(self, mock_evaluate_metrics):
        """Test evaluating a RAG interaction."""
        # Mock the evaluate_metrics method
        mock_evaluate_metrics.return_value = {
            "contextual_relevancy": 0.8,
            "faithfulness": 0.9
        }
        
        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_metrics.return_value = {
            "contextual_relevancy": 0.8,
            "faithfulness": 0.9
        }
        mock_evaluator.create_test_case.return_value = MagicMock()
        
        agent = EvaluationAgent(evaluator=mock_evaluator)
        
        result = agent.evaluate_interaction(
            query="What is RAG?",
            answer="RAG stands for Retrieval-Augmented Generation.",
            retrieved_context=[{"text": "RAG is a technique."}]
        )
        
        assert mock_evaluator.add_test_case.called
        assert mock_evaluator.evaluate_metrics.called
        assert "query" in result
        assert "answer" in result
        assert "metrics" in result