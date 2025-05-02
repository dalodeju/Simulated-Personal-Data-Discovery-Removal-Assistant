"""
Data Analyzer Agent for classifying and understanding sensitive data.
Uses NLP techniques to analyze and categorize personal information.
"""

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import spacy
from transformers import pipeline
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime
from pathlib import Path
from utils.common import setup_logging, ensure_directory

@dataclass
class AnalyzerConfig:
    """Configuration for the Data Analyzer Agent"""
    model_path: str = 'models/analyzer'
    spacy_model: str = 'en_core_web_sm'
    confidence_threshold: float = 0.7
    batch_size: int = 32
    use_gpu: bool = False
    
    def __post_init__(self):
        """Ensure model path is a Path object"""
        self.model_path = Path(self.model_path)

class DataAnalyzerAgent:
    """
    Agent responsible for analyzing and classifying personal data using NLP techniques.
    Combines rule-based and ML approaches for robust classification.
    """
    
    # Default training data for initial model fitting
    DEFAULT_TRAINING_DATA = [
        # Non-sensitive examples
        ("This is a regular public post about the weather", "not_sensitive"),
        ("Check out this awesome tutorial on Python programming", "not_sensitive"),
        ("The meeting is scheduled for next Monday at 2 PM", "not_sensitive"),
        ("I love this beautiful sunny day in the park", "not_sensitive"),
        ("Just finished reading an amazing book", "not_sensitive"),
        ("The restaurant serves delicious Italian food", "not_sensitive"),
        ("Looking forward to the weekend", "not_sensitive"),
        ("Great news! Our team won the championship", "not_sensitive"),
        ("The movie was really entertaining", "not_sensitive"),
        ("New coffee shop opened downtown", "not_sensitive"),
        
        # Financial data examples
        ("My bank account number is 1234567890", "sensitive"),
        ("Credit card details: 4111-1111-1111-1111", "sensitive"),
        ("The total transaction amount is $5000", "sensitive"),
        
        # Personal identification examples
        ("My SSN is 123-45-6789", "sensitive"),
        ("Driver's license number: X12345678", "sensitive"),
        ("Passport number: AB123456", "sensitive"),
        
        # Contact information examples
        ("You can reach me at john.doe@email.com", "sensitive"),
        ("My phone number is 555-123-4567", "sensitive"),
        ("Home address: 123 Main St, Anytown, USA", "sensitive"),
        
        # Medical information examples
        ("Patient diagnosed with type 2 diabetes", "sensitive"),
        ("Prescription: 50mg medication twice daily", "sensitive"),
        ("Medical history includes heart surgery", "sensitive"),
        
        # Credential examples
        ("Username: admin Password: secretpass123", "sensitive"),
        ("API key: sk_test_123456789", "sensitive"),
        ("Login credentials for the system", "sensitive")
    ]
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.logger = setup_logging(__name__)
        
        # Initialize NLP components
        try:
            self.nlp = spacy.load(config.spacy_model)
        except OSError:
            self.logger.warning(f"Downloading {config.spacy_model} model...")
            spacy.cli.download(config.spacy_model)
            self.nlp = spacy.load(config.spacy_model)
            
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1 if not config.use_gpu else 0
        )
        
        # Load or initialize ML models
        self._load_or_initialize_models()
        
        # Define sensitive data categories and their indicators
        self.categories = {
            'financial': ['bank', 'credit', 'account', 'payment', 'transaction', '$', '€', '£'],
            'medical': ['health', 'medical', 'doctor', 'diagnosis', 'treatment', 'patient', 'prescription'],
            'personal_id': ['ssn', 'passport', 'license', 'identity', 'id', 'social security'],
            'contact': ['address', 'phone', 'email', 'contact', '@', 'tel', 'mobile'],
            'credentials': ['password', 'username', 'login', 'authentication', 'api key', 'token']
        }

    def _load_or_initialize_models(self):
        """Load existing models or initialize and train new ones"""
        ensure_directory(self.config.model_path)
        
        model_file = self.config.model_path / "classifier.joblib"
        vectorizer_file = self.config.model_path / "vectorizer.joblib"
        
        try:
            self.classifier = joblib.load(model_file)
            self.vectorizer = joblib.load(vectorizer_file)
            self.logger.info("Loaded existing models successfully")
        except (FileNotFoundError, EOFError):
            self.logger.info("Initializing and training new models with default data")
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            )
            
            # Train with default data
            texts, labels = zip(*self.DEFAULT_TRAINING_DATA)
            X = self.vectorizer.fit_transform(texts)
            self.classifier.fit(X, labels)
            
            # Save the trained models
            joblib.dump(self.classifier, model_file)
            joblib.dump(self.vectorizer, vectorizer_file)
            self.logger.info("Successfully trained and saved new models")

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text using SpaCy"""
        doc = self.nlp(text)
        return [{
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'description': spacy.explain(ent.label_)
        } for ent in doc.ents]

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text using transformer model"""
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                'label': result['label'],
                'score': result['score'],
                'is_negative': result['label'] == 'NEGATIVE'
            }
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {str(e)}")
            return {
                'label': 'UNKNOWN',
                'score': 0.5,
                'is_negative': False
            }

    def _classify_sensitivity(self, text: str) -> Tuple[str, float]:
        """Classify text sensitivity using ML model"""
        try:
            # Transform text using vectorizer
            features = self.vectorizer.transform([text])
            
            # Get prediction probabilities
            probas = self.classifier.predict_proba(features)[0]
            predicted_class = self.classifier.classes_[np.argmax(probas)]
            confidence = np.max(probas)
            
            # Add additional checks for non-sensitive content
            if predicted_class == "sensitive":
                # Check if text contains any known sensitive patterns
                has_sensitive_pattern = any(
                    pattern in text.lower()
                    for pattern in [
                        'password', 'ssn', 'credit', 'account',
                        'license', 'passport', '@', 'phone',
                        'address', 'medical', 'patient', 'prescription'
                    ]
                )
                
                if not has_sensitive_pattern:
                    # If no sensitive patterns found, reduce confidence
                    confidence *= 0.5
                    if confidence < 0.6:  # Threshold for reclassification
                        predicted_class = "not_sensitive"
                        # Boost confidence for clearly non-sensitive text
                        if len(text.split()) >= 3:  # At least 3 words
                            confidence = max(confidence + 0.2, 0.7)
            else:
                # For predicted non-sensitive text, boost confidence if no sensitive patterns
                has_sensitive_pattern = any(
                    pattern in text.lower()
                    for pattern in [
                        'password', 'ssn', 'credit', 'account',
                        'license', 'passport', '@', 'phone',
                        'address', 'medical', 'patient', 'prescription'
                    ]
                )
                if not has_sensitive_pattern:
                    confidence = max(confidence + 0.2, 0.7)
            
            return predicted_class, confidence
        except Exception as e:
            self.logger.error(f"Error in sensitivity classification: {str(e)}")
            return "unknown", 0.0

    def analyze_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of content
        Returns detailed analysis results
        """
        text = content.get('content', '')
        if not text:
            return {
                'error': 'No content provided',
                'risk_score': 0.0,
                'sensitivity': {'class': 'unknown', 'confidence': 0.0}
            }
            
        try:
            # Extract entities
            entities = self._extract_entities(text)
            
            # Analyze sentiment
            sentiment = self._analyze_sentiment(text)
            
            # Classify sensitivity
            sensitivity_class, sensitivity_conf = self._classify_sensitivity(text)
            
            # Calculate initial risk score based on sensitivity
            risk_score = sensitivity_conf if sensitivity_class == 'sensitive' else 0.0
            
            # Adjust risk score based on entities and sentiment
            if entities:
                risk_score += min(len(entities) * 0.1, 0.3)  # Up to 0.3 increase for entities
            if sentiment['is_negative']:
                risk_score += 0.2  # Increase risk for negative sentiment
                
            # Identify sensitive data categories
            categories = {}
            for category, indicators in self.categories.items():
                matches = []
                for indicator in indicators:
                    if indicator.lower() in text.lower():
                        matches.append(indicator)
                if matches:
                    categories[category] = matches
                    risk_score += 0.1  # Increase risk for each matched category
            
            # Normalize final risk score to 0-1 range
            risk_score = min(risk_score, 1.0)
            
            return {
                'content_id': content.get('id'),
                'timestamp': datetime.now().isoformat(),
                'entities': entities,
                'sentiment': sentiment,
                'sensitivity': {
                    'class': sensitivity_class,
                    'confidence': sensitivity_conf
                },
                'risk_score': risk_score,
                'categories': categories,
                'metadata': {
                    'word_count': len(text.split()),
                    'has_urls': 'http' in text.lower(),
                    'has_numbers': any(c.isdigit() for c in text),
                    'analysis_version': '2.0'
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing content: {str(e)}")
            return {
                'error': str(e),
                'risk_score': 0.0,
                'sensitivity': {'class': 'unknown', 'confidence': 0.0}
            }

    def batch_analyze(self, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze multiple content items in batch
        Returns list of analysis results
        """
        results = []
        
        for i in range(0, len(contents), self.config.batch_size):
            batch = contents[i:i + self.config.batch_size]
            batch_results = []
            
            for content in batch:
                result = self.analyze_content(content)
                batch_results.append(result)
            
            results.extend(batch_results)
            self.logger.debug(f"Processed batch {i//self.config.batch_size + 1}")
        
        return results

    def train(self, training_data: List[Tuple[str, str]], save_models: bool = True):
        """
        Train the ML models on new data
        Args:
            training_data: List of (text, label) tuples
            save_models: Whether to save the models after training
        """
        if not training_data:
            self.logger.warning("No training data provided")
            return
            
        try:
            texts, labels = zip(*training_data)
            
            # Combine with default training data for better generalization
            all_texts = list(texts) + [text for text, _ in self.DEFAULT_TRAINING_DATA]
            all_labels = list(labels) + [label for _, label in self.DEFAULT_TRAINING_DATA]
            
            # Fit vectorizer and transform texts
            X = self.vectorizer.fit_transform(all_texts)
            
            # Train classifier
            self.classifier.fit(X, all_labels)
            
            if save_models:
                # Save models
                ensure_directory(self.config.model_path)
                joblib.dump(self.classifier, 
                           self.config.model_path / "classifier.joblib")
                joblib.dump(self.vectorizer, 
                           self.config.model_path / "vectorizer.joblib")
                self.logger.info("Successfully saved updated models")
                
        except Exception as e:
            self.logger.error(f"Error during training: {str(e)}")
            raise

def main():
    """Main function to test the Data Analyzer Agent"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize agent
    config = AnalyzerConfig()
    agent = DataAnalyzerAgent(config)
    
    # Test analysis
    test_content = {
        'id': '123',
        'content': """
        Hi, my name is John Smith and my credit card number is 1234-5678-9012-3456.
        Please contact me at john@email.com or call 555-0123 for any questions.
        """
    }
    
    try:
        result = agent.analyze_content(test_content)
        print("Analysis Results:")
        print(f"Risk Score: {result['risk_score']:.2f}")
        print(f"Detected Categories: {list(result['categories'].keys())}")
        print(f"Entities Found: {len(result['entities'])}")
        print(f"Sentiment: {result['sentiment']['label']}")
    
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == '__main__':
    main() 