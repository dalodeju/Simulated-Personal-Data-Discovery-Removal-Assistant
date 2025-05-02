"""
data analyzer agent for classifying and understanding sensitive data.
this module uses nlp techniques to analyze and categorize personal information.
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
    """
    configuration for the data analyzer agent.
    """
    model_path: str = 'models/analyzer'
    spacy_model: str = 'en_core_web_sm'
    confidence_threshold: float = 0.7
    batch_size: int = 32
    use_gpu: bool = False
    
    def __post_init__(self):
        self.model_path = Path(self.model_path)

class DataAnalyzerAgent:
    """
    agent for analyzing and classifying personal data using nlp and machine learning.
    """
    DEFAULT_TRAINING_DATA = [
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
        ("My bank account number is 1234567890", "sensitive"),
        ("Credit card details: 4111-1111-1111-1111", "sensitive"),
        ("The total transaction amount is $5000", "sensitive"),
        ("My SSN is 123-45-6789", "sensitive"),
        ("Driver's license number: X12345678", "sensitive"),
        ("Passport number: AB123456", "sensitive"),
        ("You can reach me at john.doe@email.com", "sensitive"),
        ("My phone number is 555-123-4567", "sensitive"),
        ("Home address: 123 Main St, Anytown, USA", "sensitive"),
        ("Patient diagnosed with type 2 diabetes", "sensitive"),
        ("Prescription: 50mg medication twice daily", "sensitive"),
        ("Medical history includes heart surgery", "sensitive"),
        ("Username: admin Password: secretpass123", "sensitive"),
        ("API key: sk_test_123456789", "sensitive"),
        ("Login credentials for the system", "sensitive")
    ]
    
    def __init__(self, config: AnalyzerConfig):
        self.config = config
        self.logger = setup_logging(__name__)
        try:
            self.nlp = spacy.load(config.spacy_model)
        except OSError:
            self.logger.warning(f"downloading {config.spacy_model} model...")
            spacy.cli.download(config.spacy_model)
            self.nlp = spacy.load(config.spacy_model)
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1 if not config.use_gpu else 0
        )
        self._load_or_initialize_models()
        self.categories = {
            'financial': ['bank', 'credit', 'account', 'payment', 'transaction', '$', '€', '£'],
            'medical': ['health', 'medical', 'doctor', 'diagnosis', 'treatment', 'patient', 'prescription'],
            'personal_id': ['ssn', 'passport', 'license', 'identity', 'id', 'social security'],
            'contact': ['address', 'phone', 'email', 'contact', '@', 'tel', 'mobile'],
            'credentials': ['password', 'username', 'login', 'authentication', 'api key', 'token']
        }

    def _load_or_initialize_models(self):
        """
        load classifier and vectorizer models if available, otherwise train new models.
        """
        ensure_directory(self.config.model_path)
        model_file = self.config.model_path / "classifier.joblib"
        vectorizer_file = self.config.model_path / "vectorizer.joblib"
        try:
            self.classifier = joblib.load(model_file)
            self.vectorizer = joblib.load(vectorizer_file)
            self.logger.info("loaded existing models successfully")
        except (FileNotFoundError, EOFError):
            self.logger.info("training new models with default data")
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
            texts, labels = zip(*self.DEFAULT_TRAINING_DATA)
            X = self.vectorizer.fit_transform(texts)
            self.classifier.fit(X, labels)
            joblib.dump(self.classifier, model_file)
            joblib.dump(self.vectorizer, vectorizer_file)
            self.logger.info("trained and saved new models")

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        extract named entities from text using spacy.
        """
        doc = self.nlp(text)
        return [{
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char,
            'description': spacy.explain(ent.label_)
        } for ent in doc.ents]

    def _analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        perform sentiment analysis on the input text.
        """
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                'label': result['label'],
                'score': result['score'],
                'is_negative': result['label'] == 'NEGATIVE'
            }
        except Exception as e:
            self.logger.error(f"sentiment analysis failed: {str(e)}")
            return {
                'label': 'UNKNOWN',
                'score': 0.5,
                'is_negative': False
            }

    def _classify_sensitivity(self, text: str) -> Tuple[str, float]:
        """
        classify the sensitivity of the input text using the trained model.
        returns the predicted class and confidence score.
        """
        try:
            features = self.vectorizer.transform([text])
            probas = self.classifier.predict_proba(features)[0]
            predicted_class = self.classifier.classes_[np.argmax(probas)]
            confidence = np.max(probas)
            if predicted_class == "sensitive":
                has_sensitive_pattern = any(
                    pattern in text.lower()
                    for pattern in [
                        'password', 'ssn', 'credit', 'account',
                        'license', 'passport', '@', 'phone',
                        'address', 'medical', 'patient', 'prescription'
                    ]
                )
                if not has_sensitive_pattern:
                    confidence *= 0.5
                    if confidence < 0.6:
                        predicted_class = "not_sensitive"
                        if len(text.split()) >= 3:
                            confidence = max(confidence + 0.2, 0.7)
            else:
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
            self.logger.error(f"error in sensitivity classification: {str(e)}")
            return "unknown", 0.0

    def analyze_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        analyze a content dictionary and return extracted entities, sentiment, sensitivity, risk score, and categories.
        """
        text = content.get('content', '')
        if not text:
            return {
                'error': 'no content provided',
                'risk_score': 0.0,
                'sensitivity': {'class': 'unknown', 'confidence': 0.0}
            }
        try:
            entities = self._extract_entities(text)
            sentiment = self._analyze_sentiment(text)
            sensitivity_class, sensitivity_conf = self._classify_sensitivity(text)
            risk_score = sensitivity_conf if sensitivity_class == 'sensitive' else 0.0
            if entities:
                risk_score += min(len(entities) * 0.1, 0.3)
            if sentiment['is_negative']:
                risk_score += 0.2
            categories = {}
            for category, indicators in self.categories.items():
                matches = []
                for indicator in indicators:
                    if indicator.lower() in text.lower():
                        matches.append(indicator)
                if matches:
                    categories[category] = matches
                    risk_score += 0.1
            risk_score = min(risk_score, 1.0)
            return {
                'entities': entities,
                'sentiment': sentiment,
                'sensitivity': {'class': sensitivity_class, 'confidence': sensitivity_conf},
                'risk_score': risk_score,
                'categories': categories
            }
        except Exception as e:
            self.logger.error(f"error analyzing content: {str(e)}")
            return {
                'error': str(e),
                'risk_score': 0.0,
                'sensitivity': {'class': 'unknown', 'confidence': 0.0}
            }

    def batch_analyze(self, contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        analyze a batch of content dictionaries.
        """
        return [self.analyze_content(content) for content in contents]

    def train(self, training_data: List[Tuple[str, str]], save_models: bool = True):
        """
        train the classifier and vectorizer with new training data.
        """
        texts, labels = zip(*training_data)
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        if save_models:
            model_file = self.config.model_path / "classifier.joblib"
            vectorizer_file = self.config.model_path / "vectorizer.joblib"
            joblib.dump(self.classifier, model_file)
            joblib.dump(self.vectorizer, vectorizer_file)
            self.logger.info("models saved after training")

def main():
    """
    test the data analyzer agent with sample data.
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    config = AnalyzerConfig()
    analyzer = DataAnalyzerAgent(config)
    test_content = {
        'content': "My SSN is 123-45-6789 and my email is john.doe@email.com."
    }
    result = analyzer.analyze_content(test_content)
    print("analysis result:", result)

if __name__ == '__main__':
    main() 