"""
risk evaluation agent for assessing the risk of discovered personal data.
this module uses multiple factors to compute a risk score for each data item.
"""

import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils.common import setup_logging, load_config

@dataclass
class RiskEvaluatorConfig:
    """
    configuration for the risk evaluator agent.
    """
    risk_thresholds: Dict[str, float] = None
    weights: Dict[str, float] = None
    time_decay_factor: float = 0.1
    batch_size: int = 32
    
    def __post_init__(self):
        if self.risk_thresholds is None:
            self.risk_thresholds = {
                'low': 0.3,
                'medium': 0.6,
                'high': 0.8,
                'critical': 0.9
            }
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
    agent for evaluating the risk of personal data using sensitivity, exposure, context, and other factors.
    """
    def __init__(self, config: RiskEvaluatorConfig):
        self.config = config
        self.logger = setup_logging(__name__)
        self.scaler = MinMaxScaler()
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

    def _calculate_sensitivity_score(self, data_categories: List[str], confidence_scores: Dict[str, float]) -> float:
        """
        calculate the sensitivity score based on data categories and confidence scores.
        """
        if not data_categories:
            return 0.0
        weighted_scores = []
        for category in data_categories:
            base_weight = self.sensitivity_factors.get(category, 0.3)
            confidence = confidence_scores.get(category, 0.5)
            weighted_scores.append(base_weight * confidence)
        return np.mean(weighted_scores)

    def _calculate_exposure_score(self, visibility: str, platform_reach: int, share_count: int) -> float:
        """
        calculate the exposure score based on visibility, platform reach, and share count.
        """
        base_exposure = self.exposure_factors.get(visibility.lower(), 0.5)
        reach_factor = min(platform_reach / 1000000, 1.0)
        share_factor = min(share_count / 100, 1.0)
        return base_exposure * (0.6 + 0.2 * reach_factor + 0.2 * share_factor)

    def _calculate_freshness_score(self, timestamp: Optional[str]) -> float:
        """
        calculate the freshness score based on the age of the data.
        """
        if not timestamp:
            return 0.5
        try:
            data_time = datetime.fromisoformat(timestamp)
            age_days = (datetime.now() - data_time).days
            return np.exp(-self.config.time_decay_factor * age_days)
        except (ValueError, TypeError) as e:
            self.logger.warning(f"bad timestamp: {e}")
            return 0.5

    def _calculate_combination_score(self, data_categories: List[str]) -> float:
        """
        calculate the risk score for combinations of data categories.
        """
        if len(data_categories) < 2:
            return 0.0
        combination_risk = 0.0
        for cat1 in data_categories:
            for cat2 in data_categories:
                if cat1 < cat2:
                    penalty = max(
                        self.combination_penalties.get((cat1, cat2), 0.0),
                        self.combination_penalties.get((cat2, cat1), 0.0)
                    )
                    combination_risk += penalty
        return min(combination_risk, 1.0)

    def _calculate_context_score(self, sentiment: str, is_public: bool, has_pii: bool) -> float:
        """
        calculate the context score based on sentiment, visibility, and presence of pii.
        """
        base_score = 0.0
        if isinstance(sentiment, str) and sentiment.lower() == 'negative':
            base_score += 0.3
        elif isinstance(sentiment, dict):
            if sentiment.get('label', '').lower() == 'negative':
                base_score += 0.3 * sentiment.get('score', 1.0)
        if is_public:
            base_score += 0.4
        if has_pii:
            base_score += 0.3
        return min(base_score, 1.0)

    def evaluate_risk(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        evaluate the risk of a data item and return risk score, level, and recommendations.
        """
        if not isinstance(data_item, dict):
            return self._create_error_response("bad input: data_item must be a dict")
        try:
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
            sensitivity_score = self._calculate_sensitivity_score(categories, confidence_scores)
            exposure_score = self._calculate_exposure_score(visibility, platform_reach, share_count)
            freshness_score = self._calculate_freshness_score(timestamp)
            combination_score = self._calculate_combination_score(categories)
            context_score = self._calculate_context_score(sentiment, is_public, has_pii)
            risk_score = (
                self.config.weights['sensitivity'] * sensitivity_score +
                self.config.weights['exposure'] * exposure_score +
                self.config.weights['freshness'] * freshness_score +
                self.config.weights['combination'] * combination_score +
                self.config.weights['context'] * context_score
            )
            risk_level = 'low'
            for level, threshold in sorted(self.config.risk_thresholds.items(), key=lambda x: x[1]):
                if risk_score >= threshold:
                    risk_level = level
            recommendations = self._generate_recommendations(risk_level, categories, visibility, has_pii)
            return {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'level': risk_level,
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
            return self._create_error_response(f"error evaluating risk: {str(e)}")

    def _generate_recommendations(self, risk_level: str, categories: List[str], visibility: str, has_pii: bool) -> List[str]:
        """
        generate recommendations based on risk level, categories, visibility, and pii presence.
        """
        recommendations = []
        if risk_level in ['high', 'critical']:
            recommendations.append("immediate action required to address security concerns")
            if has_pii:
                recommendations.append("remove or encrypt personally identifiable information")
            if visibility == 'public':
                recommendations.append("review and restrict visibility settings")
        category_recommendations = {
            'financial': [
                "remove or mask financial information",
                "consider using secure payment gateway instead of direct details"
            ],
            'medical': [
                "ensure medical data is properly protected",
                "review hipaa compliance requirements"
            ],
            'personal_id': [
                "remove or encrypt personal identification data",
                "use reference numbers instead of actual ids"
            ],
            'contact': [
                "consider using a contact form instead of direct details",
                "implement communication preferences management"
            ],
            'credentials': [
                "never share credential information publicly",
                "implement secure credential management system"
            ]
        }
        for category in categories:
            if category in category_recommendations:
                recommendations.extend(category_recommendations[category])
        return list(set(recommendations))

    def batch_evaluate(self, data_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        evaluate a batch of data items for risk.
        """
        return [self.evaluate_risk(item) for item in data_items]

    def _create_error_response(self, message: str) -> Dict[str, Any]:
        """
        create a standardized error response.
        """
        return {
            'error': message,
            'risk_score': 0.0,
            'risk_level': 'unknown',
            'level': 'unknown',
            'component_scores': {},
            'recommendations': [],
            'metadata': {'timestamp': datetime.now().isoformat()}
        }

def main():
    """
    test the risk evaluator agent with sample data.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    config = RiskEvaluatorConfig()
    evaluator = RiskEvaluatorAgent(config)
    test_item = {
        'categories': ['financial', 'personal_id'],
        'confidence_scores': {'financial': 0.9, 'personal_id': 0.8},
        'visibility': 'public',
        'platform_reach': 5000,
        'share_count': 10,
        'timestamp': datetime.now().isoformat(),
        'sentiment': {'label': 'NEGATIVE', 'score': 0.8}
    }
    result = evaluator.evaluate_risk(test_item)
    print("risk evaluation result:", result)

if __name__ == '__main__':
    main() 