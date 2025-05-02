"""Integration tests for the personal data discovery system"""

import pytest
import yaml
from src.agents.web_scraper import WebScraperAgent, ScraperConfig
from src.agents.data_analyzer import DataAnalyzerAgent, AnalyzerConfig
from src.agents.risk_evaluator import RiskEvaluatorAgent, RiskEvaluatorConfig
from environment.ecosystem import DigitalEcosystem, EcosystemConfig

class TestIntegration:
    @pytest.fixture
    def setup(self):
        """Set up test environment and agents"""
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            
        ecosystem = DigitalEcosystem(EcosystemConfig())
        scraper = WebScraperAgent(
            ecosystem=ecosystem,
            config=ScraperConfig(**config['web_scraper'])
        )
        analyzer = DataAnalyzerAgent(
            config=AnalyzerConfig(**config['data_analyzer'])
        )
        risk_evaluator = RiskEvaluatorAgent(
            config=RiskEvaluatorConfig(**config['risk_evaluator'])
        )
        
        return {
            'ecosystem': ecosystem,
            'scraper': scraper,
            'analyzer': analyzer,
            'risk_evaluator': risk_evaluator
        }

    def test_basic_workflow(self, setup):
        """Test basic workflow from scraping to risk evaluation"""
        # Test data
        test_text = """
        Contact: john@email.com
        Phone: 123-456-7890
        SSN: 123-45-6789
        """
        
        # 1. Discover personal data
        discovered = setup['scraper']._extract_personal_data(test_text)
        assert discovered, "Should find personal data"
        assert 'email' in discovered
        assert 'phone' in discovered
        assert 'ssn' in discovered
        
        # 2. Analyze discovered data
        analysis = setup['analyzer'].analyze_content({
            'content': test_text,
            'discovered': discovered
        })
        assert analysis, "Should produce analysis"
        assert 'risk_score' in analysis
        
        # 3. Evaluate risk
        risk_assessment = setup['risk_evaluator'].evaluate_risk(analysis)
        assert risk_assessment, "Should produce risk assessment"
        assert 'level' in risk_assessment
        assert risk_assessment['level'] in ['low', 'medium', 'high'] 