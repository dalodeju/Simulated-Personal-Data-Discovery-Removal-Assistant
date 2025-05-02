import unittest
from environment.ecosystem import DigitalEcosystem, EcosystemConfig
from src.agents.web_scraper import WebScraperAgent, ScraperConfig
from src.agents.data_analyzer import DataAnalyzerAgent, AnalyzerConfig
from src.agents.risk_evaluator import RiskEvaluatorAgent, RiskEvaluatorConfig
from src.agents.recommender import RecommendationAgent

class TestBasicAgentFunctionality(unittest.TestCase):
    """Test basic functionality of all agents."""
    
    def setUp(self):
        """Set up test environment and agents."""
        self.ecosystem = DigitalEcosystem(EcosystemConfig())
        
        # Initialize agents
        self.web_scraper = WebScraperAgent(
            ecosystem=self.ecosystem,
            config=ScraperConfig()
        )
        
        self.data_analyzer = DataAnalyzerAgent(
            config=AnalyzerConfig()
        )
        
        self.risk_evaluator = RiskEvaluatorAgent(
            config=RiskEvaluatorConfig()
        )
        
        self.recommender = RecommendationAgent()
    
    def test_environment_initialization(self):
        """Test that environment is properly initialized."""
        profiles = self.ecosystem.search_profiles("")
        self.assertIsNotNone(profiles)
        self.assertIsInstance(profiles, list)
    
    def test_web_scraper_functionality(self):
        """Test basic web scraper functionality."""
        test_text = """
        Contact: john@email.com
        Phone: 123-456-7890
        SSN: 123-45-6789
        """
        
        # Test extraction
        results = self.web_scraper._extract_personal_data(test_text)
        self.assertIn('email', results)
        self.assertIn('phone', results)
        self.assertIn('ssn', results)
        
        # Test confidence calculation
        confidence = self.web_scraper._calculate_confidence(test_text, 'email')
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
    
    def test_data_analyzer_functionality(self):
        """Test basic data analyzer functionality."""
        test_content = {
            'content': 'This is a test with sensitive information: test@email.com',
            'discovered': {'email': ['test@email.com']}
        }
        
        results = self.data_analyzer.analyze_content(test_content)
        self.assertIsNotNone(results)
        self.assertIn('risk_score', results)
    
    def test_risk_evaluator_functionality(self):
        """Test basic risk evaluator functionality."""
        test_data = {
            'content_id': '123',
            'risk_score': 0.8,
            'categories': {'personal_id': ['ssn']}
        }
        
        results = self.risk_evaluator.evaluate_risk(test_data)
        self.assertIsNotNone(results)
        self.assertIn('level', results)
    
    def test_recommender_functionality(self):
        """Test basic recommender functionality."""
        test_assessment = {
            'level': 'high',
            'details': {
                'concerns': ['SSN exposed'],
                'stats': {'high_sensitivity_items': 1}
            }
        }
        
        results = self.recommender.generate(test_assessment)
        self.assertIsNotNone(results)
        self.assertIn('priority_actions', results)
        self.assertIn('protection_measures', results)

if __name__ == '__main__':
    unittest.main() 