"""
recommendation agent for suggesting data protection and removal actions.
this module generates recommendations based on risk assessment results.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

# action templates by risk level. not exhaustive, but covers the basics
ACTIONS = {
    'high': [
        "immediately remove {data_type} from {location}",
        "encrypt all instances of {data_type}",
        "review and restrict access to {data_type}"
    ],
    'medium': [
        "consider removing {data_type} if not needed",
        "add access controls to {data_type}",
        "monitor usage of {data_type}"
    ],
    'low': [
        "regularly review {data_type} storage",
        "document {data_type} usage",
        "update {data_type} handling policy"
    ]
}

# specific actions for different types of data. could add more if you want
DATA_ACTIONS = {
    'ssn': [
        "remove ssn immediately",
        "use last 4 digits only if needed",
        "encrypt ssn in storage"
    ],
    'credit_card': [
        "remove full credit card number",
        "use last 4 digits only",
        "implement secure payment system"
    ],
    'email': [
        "use contact form instead of displaying email",
        "add spam protection",
        "monitor for unauthorized access"
    ],
    'phone': [
        "consider using a contact form",
        "limit phone number visibility",
        "monitor for abuse"
    ]
}

class RecommendationAgent:
    """
    agent for generating recommendations for data protection and removal based on risk assessment.
    """
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("setting up recommendation agent...")
        self.actions = ACTIONS
        self.type_actions = DATA_ACTIONS

    def generate(self, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        generate recommendations based on the provided risk assessment.
        returns a dictionary with actions, measures, and a summary.
        """
        try:
            if not isinstance(risk_assessment, dict):
                raise ValueError("bad risk assessment format")
            risk_level = risk_assessment.get('level', 'low')
            details = risk_assessment.get('details', {})
            recs = {
                'timestamp': datetime.now().isoformat(),
                'risk_level': risk_level,
                'priority_actions': [],
                'protection_measures': [],
                'summary': {
                    'total_recommendations': 0,
                    'priority_level': risk_level
                }
            }
            # start with general stuff based on risk
            recs['priority_actions'].extend(self._get_general_recommendations(risk_level))
            # add specific stuff based on what we found
            concerns = details.get('concerns', [])
            specific_recs = self._get_specific_recommendations(concerns)
            recs['protection_measures'].extend(specific_recs)
            # add recommendations for data types
            if 'stats' in details:
                type_recs = self._get_type_recommendations(details['stats'])
                recs['protection_measures'].extend(type_recs)
            # clean up and count
            recs['priority_actions'] = list(set(recs['priority_actions']))
            recs['protection_measures'] = list(set(recs['protection_measures']))
            total = len(recs['priority_actions']) + len(recs['protection_measures'])
            recs['summary']['total_recommendations'] = total
            self.logger.info(f"generated {total} recommendations for {risk_level} risk")
            self.logger.debug(f"priority actions: {len(recs['priority_actions'])}, protection measures: {len(recs['protection_measures'])}")
            return recs
        except Exception as e:
            self.logger.error(f"couldn't generate recommendations: {str(e)}")
            return self._empty_recommendations(str(e))

    def _get_general_recommendations(self, risk_level: str) -> List[str]:
        """
        get general recommendations based on risk level.
        """
        templates = self.actions.get(risk_level, self.actions['low'])
        return [
            template.format(
                data_type="personal data",
                location="the system"
            )
            for template in templates
        ]

    def _get_specific_recommendations(self, concerns: List[str]) -> List[str]:
        """
        get specific recommendations based on identified concerns.
        """
        recs = []
        for concern in concerns:
            if 'ssn' in concern.lower():
                recs.extend(self.type_actions['ssn'])
            elif 'credit card' in concern.lower():
                recs.extend(self.type_actions['credit_card'])
            elif 'multiple types' in concern.lower():
                recs.append("implement comprehensive data protection policy")
        return list(set(recs))

    def _get_type_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """
        get recommendations based on data type statistics.
        """
        recs = []
        if stats.get('high_sensitivity_items', 0) > 0:
            recs.extend([
                "implement data encryption",
                "set up access logging",
                "create data removal procedure"
            ])
        return recs

    def _empty_recommendations(self, error_msg=None):
        """
        return an empty recommendations structure in case of error.
        """
        return {
            'timestamp': datetime.now().isoformat(),
            'risk_level': 'unknown',
            'priority_actions': [],
            'protection_measures': [],
            'summary': {
                'total_recommendations': 0,
                'priority_level': 'unknown'
            },
            'error': error_msg
        }

def main():
    """
    test the recommendation agent with sample data.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    recommender = RecommendationAgent()
    
    # Fake a risk assessment
    test_assessment = {
        'level': 'high',
        'details': {
            'concerns': [
                "Social Security Numbers exposed",
                "Multiple types of personal data found together"
            ],
            'stats': {
                'high_sensitivity_items': 2
            }
        }
    }
    
    # Generate recommendations
    results = recommender.generate(test_assessment)
    
    # Show what we got
    print("\nRecommendation Results:")
    print(f"Risk Level: {results['risk_level']}")
    print(f"Total Recommendations: {results['summary']['total_recommendations']}")
    
    print("\nPriority Actions:")
    for action in results['priority_actions']:
        print(f"- {action}")
        
    print("\nProtection Measures:")
    for measure in results['protection_measures']:
        print(f"- {measure}")

if __name__ == '__main__':
    main() 