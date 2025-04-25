"""Tests for the WebScraperAgent class."""

import unittest
from unittest.mock import patch, MagicMock
from src.agents.web_scraper import WebScraperAgent, ScraperConfig
from src.environment.ecosystem import DigitalEcosystem

class TestWebScraperAgent(unittest.TestCase):
    """Test cases for WebScraperAgent functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = ScraperConfig()
        self.ecosystem = MagicMock(spec=DigitalEcosystem)
        self.agent = WebScraperAgent(self.ecosystem, self.config)
        
    def test_extract_personal_data_empty(self):
        """Test personal data extraction with empty input."""
        results = self.agent._extract_personal_data("")
        self.assertEqual(results, {})
        
        results = self.agent._extract_personal_data(None)
        self.assertEqual(results, {})
        
    def test_extract_personal_data_no_matches(self):
        """Test extraction with text containing no personal data."""
        text = "This is a regular text with no personal information."
        results = self.agent._extract_personal_data(text)
        self.assertEqual(results, {})
        
    def test_extract_personal_data_basic(self):
        """Test basic personal data extraction."""
        text = "Contact me at test@email.com or 123-456-7890"
        results = self.agent._extract_personal_data(text)
        self.assertIn('email', results)
        self.assertIn('phone', results)
        self.assertEqual(results['email'], ['test@email.com'])
        self.assertEqual(results['phone'], ['123-456-7890'])
        
    def test_extract_personal_data_multiple(self):
        """Test extraction of multiple instances of personal data."""
        text = """
        Email: test1@email.com
        Alt Email: test2@email.com
        Phone: 123-456-7890
        Alt Phone: 098-765-4321
        """
        results = self.agent._extract_personal_data(text)
        self.assertEqual(len(results['email']), 2)
        self.assertEqual(len(results['phone']), 2)
        
    def test_calculate_confidence_empty(self):
        """Test confidence calculation with empty input."""
        confidence = self.agent._calculate_confidence("", "email")
        self.assertEqual(confidence, 0.0)
        
    def test_calculate_confidence_invalid_type(self):
        """Test confidence calculation with invalid data type."""
        confidence = self.agent._calculate_confidence("test@email.com", "invalid_type")
        self.assertEqual(confidence, 0.0)
        
    def test_calculate_confidence_basic(self):
        """Test basic confidence score calculation."""
        text = "Email me at test@email.com"
        confidence = self.agent._calculate_confidence(text, "email")
        self.assertGreater(confidence, 0.5)
        
    def test_calculate_confidence_context(self):
        """Test confidence calculation with context words."""
        text = "For contact, email me at test@email.com"
        confidence = self.agent._calculate_confidence(text, "email")
        self.assertGreater(confidence, 0.7)
        
    @patch('src.environment.ecosystem.DigitalEcosystem.search_profiles')
    def test_discover_profile_data_empty(self, mock_search):
        """Test profile discovery with empty query."""
        results = self.agent.discover_profile_data("")
        self.assertEqual(results, [])
        
        results = self.agent.discover_profile_data(None)
        self.assertEqual(results, [])
        
    @patch('src.environment.ecosystem.DigitalEcosystem.search_profiles')
    def test_discover_profile_data_basic(self, mock_search):
        """Test basic profile data discovery."""
        mock_profile = {
            'id': '123',
            'email': 'test@example.com',
            'phone': '123-456-7890',
            'visibility': 'public'  # Add visibility for confidence calculation
        }
        self.ecosystem.search_profiles.return_value = [mock_profile]
        self.ecosystem.search_profiles.side_effect = None  # Ensure no simulated errors
        
        results = self.agent.discover_profile_data("test query")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['profile_id'], '123')
        self.assertIn('personal_data', results[0])
        self.assertIn('confidence_scores', results[0])
        self.assertIn('profile_metadata', results[0])
        
        # Verify extracted data
        personal_data = results[0]['personal_data']
        self.assertIn('email', personal_data)
        self.assertIn('phone', personal_data)
        self.assertEqual(personal_data['email'], ['test@example.com'])
        self.assertEqual(personal_data['phone'], ['123-456-7890'])
        
    @patch('src.environment.ecosystem.DigitalEcosystem.search_profiles')
    def test_discover_profile_data_no_matches(self, mock_search):
        """Test profile discovery with no matching profiles."""
        mock_search.return_value = []
        results = self.agent.discover_profile_data("no matches")
        self.assertEqual(results, [])
        
    def test_retry_operation_success(self):
        """Test successful retry operation."""
        counter = {'attempts': 0}
        
        def operation():
            counter['attempts'] += 1
            return "success"
            
        result = self.agent._retry_operation(operation)
        self.assertEqual(result, "success")
        self.assertEqual(counter['attempts'], 1)
        
    def test_retry_operation_failure(self):
        """Test retry operation that always fails."""
        def operation():
            raise ValueError("Test error")
            
        with self.assertRaises(ValueError):
            self.agent._retry_operation(operation)
            
    def test_retry_operation_eventual_success(self):
        """Test retry operation that succeeds after failures."""
        counter = {'attempts': 0}
        
        def operation():
            counter['attempts'] += 1
            if counter['attempts'] < 2:
                raise ValueError("Test error")
            return "success"
            
        result = self.agent._retry_operation(operation)
        self.assertEqual(result, "success")
        self.assertEqual(counter['attempts'], 2)
        
    def test_validate_format(self):
        """Test format validation for different data types."""
        # Test email validation
        self.assertTrue(self.agent._validate_format("test@example.com", "email"))
        self.assertFalse(self.agent._validate_format("invalid-email", "email"))
        
        # Test phone validation
        self.assertTrue(self.agent._validate_format("123-456-7890", "phone"))
        self.assertFalse(self.agent._validate_format("123-456", "phone"))
        
        # Test credit card validation (using test numbers)
        self.assertTrue(self.agent._validate_format("4532015112830366", "credit_card"))  # Valid Visa
        self.assertFalse(self.agent._validate_format("1234567890", "credit_card"))
        
        # Test IP address validation
        self.assertTrue(self.agent._validate_format("192.168.1.1", "ip_address"))
        self.assertFalse(self.agent._validate_format("256.256.256.256", "ip_address"))

if __name__ == '__main__':
    unittest.main() 