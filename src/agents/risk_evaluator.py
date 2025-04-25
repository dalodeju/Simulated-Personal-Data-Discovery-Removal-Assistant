"""
Risk Evaluation Agent for assessing the risk level of discovered personal data.
Uses multiple factors to calculate comprehensive risk scores.
"""

import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.utils.common import setup_logging, load_config

@dataclass
class RiskEvaluatorConfig:
    """Configuration for the Risk Evaluator Agent"""
    risk_thresholds: Dict[str, float] = None
    weights: Dict[str, float] = None
    time_decay_factor: float = 0.1
    batch_size: int = 32
    
    def __post_init__(self):
        # Default risk thresholds if none provided
        if self.risk_thresholds is None:
            self.risk_thresholds = {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8,
                'critical': 0.9
            }
        
        # Default feature weights if none provided
        if self.weights is None:
            self.weights = {
                'sensitivity': 0.35,
                'exposure': 0.25,
                'freshness': 0.15,
                'combination': 0.15,
                'context': 0.10
            }

class RiskEvaluatorAgent:
    """
    Agent responsible for evaluating the risk level of discovered personal data.
    Considers multiple factors including data sensitivity, exposure, and context.
    """
    
    def __init__(self, config: RiskEvaluatorConfig):
        self.config = config
        self.logger = setup_logging(__name__)
        self.scaler = MinMaxScaler()
        
        # Risk factors and their relative weights within categories
        self.sensitivity_factors = {
            'financial': 0.9,
            'medical': 0.85,
            'personal_id': 0.8,
            'contact': 0.6,
            'credentials': 0.75,
            'location': 0.5,
            'demographic': 0.4
        }
        
        self.exposure_factors = {
            'public': 1.0,
            'friends': 0.7,
            'private': 0.3,
            'encrypted': 0.1
        }
        
        self.combination_penalties = {
            ('financial', 'personal_id'): 0.3,
            ('medical', 'personal_id'): 0.3,
            ('contact', 'financial'): 0.2,
            ('location', 'contact'): 0.2,
            ('medical', 'contact'): 0.25,
            ('financial', 'credentials'): 0.35,
            ('personal_id', 'credentials'): 0.4
        }

    def _calculate_sensitivity_score(self, data_categories: List[str], 
                                  confidence_scores: Dict[str, float]) -> float:
        """Calculate sensitivity score based on data categories and confidence"""
        if not data_categories:
            return 0.0
            
        weighted_scores = []
        for category in data_categories:
            base_weight = self.sensitivity_factors.get(category, 0.3)
            confidence = confidence_scores.get(category, 0.5)
            weighted_scores.append(base_weight * confidence)
            
        return np.mean(weighted_scores)

    def _calculate_exposure_score(self, visibility: str, 
                                platform_reach: int,
                                share_count: int) -> float:
        """Calculate exposure score based on visibility and reach"""
        # Base exposure from visibility level
        base_exposure = self.exposure_factors.get(visibility.lower(), 0.5)
        
        # Adjust for platform reach (normalized to 0-1 range)
        reach_factor = min(platform_reach / 1000000, 1.0)
        
        # Adjust for sharing activity (normalized to 0-1 range)
        share_factor = min(share_count / 100, 1.0)
        
        return base_exposure * (0.6 + 0.2 * reach_factor + 0.2 * share_factor)

    def _calculate_freshness_score(self, timestamp: Optional[str]) -> float:
        """Calculate freshness score based on data age"""
        if not timestamp:
            return 0.5  # Default score if no timestamp
            
        try:
            data_time = datetime.fromisoformat(timestamp)
            age_days = (datetime.now() - data_time).days
            
            # Exponential decay based on age
            return np.exp(-self.config.time_decay_factor * age_days)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Invalid timestamp format: {e}")
            return 0.5  # Default score if timestamp is invalid

    def _calculate_combination_score(self, 
                                  data_categories: List[str]) -> float:
        """Calculate risk score increase from dangerous category combinations"""
        if len(data_categories) < 2:
            return 0.0
            
        combination_risk = 0.0
        for cat1 in data_categories:
            for cat2 in data_categories:
                if cat1 < cat2:  # Avoid counting pairs twice
                    # Check both orderings of the pair
                    penalty = max(
                        self.combination_penalties.get((cat1, cat2), 0.0),
                        self.combination_penalties.get((cat2, cat1), 0.0)
                    )
                    combination_risk += penalty
                    
        return min(combination_risk, 1.0)

    def _calculate_context_score(self, 
                               sentiment: str,
                               is_public: bool,
                               has_pii: bool) -> float:
        """Calculate context-based risk score"""
        base_score = 0.0
        
        # Adjust for sentiment
        if isinstance(sentiment, str) and sentiment.lower() == 'negative':
            base_score += 0.3
        elif isinstance(sentiment, dict):
            # Handle sentiment analysis results
            if sentiment.get('label', '').lower() == 'negative':
                base_score += 0.3 * sentiment.get('score', 1.0)
        
        # Adjust for public visibility
        if is_public:
            base_score += 0.4
            
        # Adjust for presence of PII
        if has_pii:
            base_score += 0.3
            
        return min(base_score, 1.0)

    def evaluate_risk(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the risk level of a single data item
        Returns detailed risk assessment
        """
        if not isinstance(data_item, dict):
            return self._create_error_response("Invalid input: data_item must be a dictionary")
            
        try:
            # Extract relevant features with proper type checking
            categories = data_item.get('categories', [])
            if not isinstance(categories, list):
                categories = []
                
            confidence_scores = data_item.get('confidence_scores', {})
            if not isinstance(confidence_scores, dict):
                confidence_scores = {}
                
            visibility = str(data_item.get('visibility', 'private')).lower()
            platform_reach = int(data_item.get('platform_reach', 1000))
            share_count = int(data_item.get('share_count', 0))
            timestamp = data_item.get('timestamp')
            sentiment = data_item.get('sentiment', {})
            is_public = visibility == 'public'
            has_pii = any(cat in self.sensitivity_factors for cat in categories)
            
            # Calculate component scores
            sensitivity_score = self._calculate_sensitivity_score(
                categories, confidence_scores
            )
            exposure_score = self._calculate_exposure_score(
                visibility, platform_reach, share_count
            )
            freshness_score = self._calculate_freshness_score(timestamp)
            combination_score = self._calculate_combination_score(categories)
            context_score = self._calculate_context_score(
                sentiment, is_public, has_pii
            )
            
            # Calculate weighted risk score
            risk_score = (
                self.config.weights['sensitivity'] * sensitivity_score +
                self.config.weights['exposure'] * exposure_score +
                self.config.weights['freshness'] * freshness_score +
                self.config.weights['combination'] * combination_score +
                self.config.weights['context'] * context_score
            )
            
            # Determine risk level
            risk_level = 'low'
            for level, threshold in sorted(
                self.config.risk_thresholds.items(),
                key=lambda x: x[1]
            ):
                if risk_score >= threshold:
                    risk_level = level
            
            # Build recommendations based on findings
            recommendations = self._generate_recommendations(
                risk_level, categories, visibility, has_pii
            )
            
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'level': risk_level,  # For backward compatibility
                'component_scores': {
                    'sensitivity': sensitivity_score,
                    'exposure': exposure_score,
                    'freshness': freshness_score,
                    'combination': combination_score,
                    'context': context_score
                },
                'recommendations': recommendations,
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'categories_analyzed': categories,
                    'visibility': visibility,
                    'has_pii': has_pii
                }
            }
            
        except Exception as e:
            return self._create_error_response(f"Error evaluating risk: {str(e)}")

    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create a standardized error response"""
        self.logger.error(error_message)
        return {
            'error': error_message,
            'risk_level': 'unknown',
            'level': 'unknown',
            'risk_score': 0.0,
            'component_scores': {
                'sensitivity': 0.0,
                'exposure': 0.0,
                'freshness': 0.0,
                'combination': 0.0,
                'context': 0.0
            },
            'recommendations': ['Unable to evaluate risk due to error'],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'error_type': 'evaluation_error'
            }
        }

    def _generate_recommendations(self, 
                                risk_level: str,
                                categories: List[str],
                                visibility: str,
                                has_pii: bool) -> List[str]:
        """Generate specific recommendations based on risk assessment"""
        recommendations = []
        
        # Risk level based recommendations
        if risk_level in ['high', 'critical']:
            recommendations.append("Immediate action required to address security concerns")
            if has_pii:
                recommendations.append("Remove or encrypt personally identifiable information")
            if visibility == 'public':
                recommendations.append("Review and restrict visibility settings")
                
        # Category specific recommendations
        category_recommendations = {
            'financial': [
                "Remove or mask financial information",
                "Consider using secure payment gateway instead of direct details"
            ],
            'medical': [
                "Ensure medical data is properly protected",
                "Review HIPAA compliance requirements"
            ],
            'personal_id': [
                "Remove or encrypt personal identification data",
                "Use reference numbers instead of actual IDs"
            ],
            'contact': [
                "Consider using a contact form instead of direct details",
                "Implement communication preferences management"
            ],
            'credentials': [
                "Never share credential information publicly",
                "Implement secure credential management system"
            ]
        }
        
        for category in categories:
            if category in category_recommendations:
                recommendations.extend(category_recommendations[category])
                
        return list(set(recommendations))  # Remove duplicates

    def batch_evaluate(self, 
                      data_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Evaluate risk for multiple data items in batch
        Returns list of risk assessments
        """
        if not isinstance(data_items, list):
            return [self._create_error_response("Invalid input: data_items must be a list")]
            
        results = []
        
        for i in range(0, len(data_items), self.config.batch_size):
            batch = data_items[i:i + self.config.batch_size]
            batch_results = []
            
            for item in batch:
                result = self.evaluate_risk(item)
                batch_results.append(result)
            
            results.extend(batch_results)
            self.logger.debug(f"Processed batch {i//self.config.batch_size + 1}")
        
        return results

def main():
    """Main function to test the Risk Evaluator Agent"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agent
    config = RiskEvaluatorConfig()
    agent = RiskEvaluatorAgent(config)
    
    # Test evaluation
    test_item = {
        'id': '123',
        'categories': ['financial', 'personal_id'],
        'confidence_scores': {
            'financial': 0.9,
            'personal_id': 0.85
        },
        'visibility': 'public',
        'platform_reach': 50000,
        'share_count': 25,
        'timestamp': datetime.now().isoformat(),
        'sentiment': {'label': 'negative'},
    }
    
    try:
        result = agent.evaluate_risk(test_item)
        print("Risk Assessment Results:")
        print(f"Risk Score: {result['risk_score']:.2f}")
        print(f"Risk Level: {result['risk_level']}")
        print("\nComponent Scores:")
        for component, score in result['component_scores'].items():
            print(f"- {component}: {score:.2f}")
        print("\nRecommendations:")
        for rec in result['recommendations']:
            print(f"- {rec}")
    
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == '__main__':
    main() 