"""Tests for the DataAnalyzerAgent class."""

import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from src.agents.data_analyzer import DataAnalyzerAgent, AnalyzerConfig

class TestDataAnalyzerAgent(unittest.TestCase):
    """Test cases for DataAnalyzerAgent functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = AnalyzerConfig()
        self.agent = DataAnalyzerAgent(self.config)
        
    def test_extract_entities_basic(self):
        """Test basic entity extraction."""
        text = "John Smith works at Microsoft in Seattle."
        entities = self.agent._extract_entities(text)
        
        # Verify we got some entities
        self.assertTrue(len(entities) > 0)
        
        # Check entity structure
        entity = entities[0]
        self.assertIn('text', entity)
        self.assertIn('label', entity)
        self.assertIn('start', entity)
        self.assertIn('end', entity)
        self.assertIn('description', entity)
        
    def test_analyze_sentiment_positive(self):
        """Test sentiment analysis with positive text."""
        text = "I love this product, it's amazing!"
        sentiment = self.agent._analyze_sentiment(text)
        
        self.assertIn('label', sentiment)
        self.assertIn('score', sentiment)
        self.assertIn('is_negative', sentiment)
        self.assertGreater(sentiment['score'], 0.5)
        
    def test_analyze_sentiment_negative(self):
        """Test sentiment analysis with negative text."""
        text = "This is terrible, I hate it."
        sentiment = self.agent._analyze_sentiment(text)
        
        self.assertIn('label', sentiment)
        self.assertIn('score', sentiment)
        self.assertIn('is_negative', sentiment)
        self.assertTrue(sentiment['is_negative'])
        
    def test_classify_sensitivity_sensitive(self):
        """Test sensitivity classification with sensitive data."""
        text = "My credit card number is 4111-1111-1111-1111"
        sensitivity_class, confidence = self.agent._classify_sensitivity(text)
        
        self.assertEqual(sensitivity_class, "sensitive")
        self.assertGreater(confidence, 0.5)
        
    def test_classify_sensitivity_not_sensitive(self):
        """Test sensitivity classification with non-sensitive data."""
        text = "The weather is nice today"
        sensitivity_class, confidence = self.agent._classify_sensitivity(text)
        
        self.assertEqual(sensitivity_class, "not_sensitive")
        self.assertGreater(confidence, 0.5)
        
    def test_analyze_content_complete(self):
        """Test complete content analysis."""
        content = {
            'id': '123',
            'content': """
            Hi, my name is John Smith and my credit card number is 4111-1111-1111-1111.
            Please contact me at john@email.com or call 555-0123 for any questions.
            """
        }
        
        result = self.agent.analyze_content(content)
        
        # Check all expected components are present
        self.assertIn('content_id', result)
        self.assertIn('timestamp', result)
        self.assertIn('entities', result)
        self.assertIn('sentiment', result)
        self.assertIn('sensitivity', result)
        self.assertIn('risk_score', result)
        self.assertIn('categories', result)
        self.assertIn('metadata', result)
        
        # Verify risk score is calculated
        self.assertGreater(result['risk_score'], 0)
        
        # Check categories
        self.assertTrue(len(result['categories']) > 0)
        
    def test_analyze_content_empty(self):
        """Test content analysis with empty input."""
        content = {'content': ''}
        result = self.agent.analyze_content(content)
        
        self.assertIn('error', result)
        self.assertEqual(result['risk_score'], 0.0)
        
    def test_batch_analyze(self):
        """Test batch analysis of multiple content items."""
        contents = [
            {'id': '1', 'content': 'Normal public post about weather'},
            {'id': '2', 'content': 'My SSN is 123-45-6789'},
            {'id': '3', 'content': 'Contact me at test@email.com'}
        ]
        
        results = self.agent.batch_analyze(contents)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('risk_score', result)
            self.assertIn('categories', result)

if __name__ == '__main__':
    unittest.main() 