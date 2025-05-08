"""
Recommendation Agent for suggesting data protection and removal actions.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

# action templates by risk level
ACTIONS = {
    'high': [
        "Immediately remove {data_type} from {location}",
        "Encrypt all instances of {data_type}",
        "Review and restrict access to {data_type}"
    ],
    'medium': [
        "Consider removing {data_type} if not needed",
        "Add access controls to {data_type}",
        "Monitor usage of {data_type}"
    ],
    'low': [
        "Regularly review {data_type} storage",
        "Document {data_type} usage",
        "Update {data_type} handling policy"
    ]
}

# specific actions for different types of data
DATA_ACTIONS = {
    'ssn': [
        "Remove SSN immediately",
        "Use last 4 digits only if needed",
        "Encrypt SSN in storage"
    ],
    'credit_card': [
        "Remove full credit card number",
        "Use last 4 digits only",
        "Implement secure payment system"
    ],
    'email': [
        "Use contact form instead of displaying email",
        "Add spam protection",
        "Monitor for unauthorized access"
    ],
    'phone': [
        "Consider using a contact form",
        "Limit phone number visibility",
        "Monitor for abuse"
    ]
}

class RecommendationAgent:
    """
    Agent responsible for generating actionable recommendations.
    Uses predefined templates based on risk levels and data types.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Setting up recommendation agent...")
        
        # Load templates
        self.actions = ACTIONS
        self.type_actions = DATA_ACTIONS

    def generate(self, risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations based on risk assessment.
        Returns prioritized list of actions and protection measures.
        """
        try:
            # Sanity check
            if not isinstance(risk_assessment, dict):
                raise ValueError("Invalid risk assessment format")
                
            risk_level = risk_assessment.get('level', 'low')
            details = risk_assessment.get('details', {})
            
            # Build our recommendations
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
            
            # Start with general stuff based on risk
            recs['priority_actions'].extend(
                self._get_general_recommendations(risk_level)
            )
            
            # Add specific stuff based on what we found
            concerns = details.get('concerns', [])
            specific_recs = self._get_specific_recommendations(concerns)
            recs['protection_measures'].extend(specific_recs)
            
            # Add recommendations for data types
            if 'stats' in details:
                type_recs = self._get_type_recommendations(details['stats'])
                recs['protection_measures'].extend(type_recs)
            
            # Clean up and count
            recs['priority_actions'] = list(set(recs['priority_actions']))
            recs['protection_measures'] = list(set(recs['protection_measures']))
            
            total = len(recs['priority_actions']) + len(recs['protection_measures'])
            recs['summary']['total_recommendations'] = total
            
            # Log what we did
            self.logger.info(
                f"Generated {total} recommendations for {risk_level} risk"
            )
            self.logger.debug(
                f"Priority actions: {len(recs['priority_actions'])}, "
                f"Protection measures: {len(recs['protection_measures'])}"
            )
            
            return recs
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendations: {str(e)}")
            return self._empty_recommendations(str(e))

    def _get_general_recommendations(self, risk_level: str) -> List[str]:
        """Get general recommendations based on risk level"""
        templates = self.actions.get(risk_level, self.actions['low'])
        return [
            template.format(
                data_type="personal data",
                location="the system"
            )
            for template in templates
        ]

    def _get_specific_recommendations(self, concerns: List[str]) -> List[str]:
        """Get recommendations for specific issues we found"""
        recs = []
        
        for concern in concerns:
            # Handle different types of concerns
            if 'SSN' in concern:
                recs.extend(self.type_actions['ssn'])
            elif 'credit card' in concern.lower():
                recs.extend(self.type_actions['credit_card'])
            elif 'multiple types' in concern.lower():
                # This is bad - need comprehensive protection
                recs.append(
                    "Implement comprehensive data protection policy"
                )
                
        return list(set(recs))  # No duplicates

    def _get_type_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Get recommendations based on types of data found"""
        recs = []
        
        # If we found sensitive stuff, add security measures
        if stats.get('high_sensitivity_items', 0) > 0:
            recs.extend([
                "Implement data encryption",
                "Set up access logging",
                "Create data removal procedure"
            ])
            
        return recs
        
    def _empty_recommendations(self, error_msg=None):
        """Return empty recommendation structure"""
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
    """Test the recommendation agent with sample data"""
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
