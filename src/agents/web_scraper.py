"""
personal data discovery using regular expressions.
this module provides functionality to identify personal data such as emails, phone numbers, and other identifiers in text.
"""

import re
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from environment.ecosystem import DigitalEcosystem, EcosystemConfig
from utils.common import setup_logging, load_config
import logging

# load configuration for the web scraper agent
default_config = load_config().get('web_scraper', {
    'max_retries': 3,
    'retry_delay': 1.0,
    'confidence_threshold': 0.7
})

# regular expression patterns for personal data types
PATTERNS = {
    'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    'phone': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
    'ssn': r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
    'credit_card': r'\b(?:\d[ -]*?){13,16}\b',
    'date_of_birth': r'\b\d{2}[-/]\d{2}[-/]\d{4}\b',
    'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    'url': r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
}

class ValidationError(Exception):
    """
    exception for invalid configuration or input values.
    """
    pass

@dataclass
class ScraperConfig:
    """
    configuration for the web scraper agent.
    """
    max_retries: int = default_config.get('max_retries', 3)
    retry_delay: float = default_config.get('retry_delay', 1.0)
    confidence_threshold: float = default_config.get('confidence_threshold', 0.7)
    
    def __post_init__(self):
        if self.max_retries < 1:
            raise ValidationError("max_retries must be at least 1")
        if self.retry_delay < 0:
            raise ValidationError("retry_delay cannot be negative")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValidationError("confidence_threshold must be between 0 and 1")

class WebScraperAgent:
    """
    agent for extracting personal data from text using regular expressions.
    """
    def __init__(self, ecosystem: Optional[DigitalEcosystem] = None, config: Optional[ScraperConfig] = None):
        super().__init__()
        self.ecosystem = ecosystem
        self.config = config or ScraperConfig()
        self.logger = logging.getLogger(__name__)
        # compile regular expressions for efficiency
        self.patterns = {
            'email': re.compile(PATTERNS['email']),
            'phone': re.compile(r'\+?1?\d{9,15}|\(\d{3}\)\s*\d{3}[-.]?\d{4}|\d{3}[-.]?\d{3}[-.]?\d{4}'),
            'ssn': re.compile(r'\d{3}-\d{2}-\d{4}'),
            'credit_card': re.compile(r'\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}'),
            'date_of_birth': re.compile(r'\d{2}[/-]\d{2}[/-]\d{4}'),
            'ip_address': re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'),
            'url': re.compile(PATTERNS['url'])
        }
        self.logger.info("web scraper agent initialized")
        
    def _extract_personal_data(self, text: str) -> Dict[str, List[str]]:
        """
        extract personal data from text using regular expressions.
        returns a dictionary mapping data types to lists of matches.
        """
        if text is None:
            self.logger.warning("input text is None")
            return {}
        if not isinstance(text, str):
            text = str(text)
            self.logger.warning(f"converted non-string input to string: {type(text)}")
        if not text.strip():
            self.logger.warning("input text is empty")
            return {}
        results = {}
        try:
            for data_type, pattern in self.patterns.items():
                matches = pattern.findall(text)
                if matches:
                    cleaned_matches = []
                    for match in matches:
                        if isinstance(match, tuple):
                            cleaned_match = '-'.join(match)
                        else:
                            cleaned_match = match.strip()
                        if cleaned_match:
                            cleaned_matches.append(cleaned_match)
                    if cleaned_matches:
                        self.logger.debug(f"found {len(cleaned_matches)} matches for {data_type}")
                        results[data_type] = cleaned_matches
            return results
        except Exception as e:
            self.logger.error(f"error extracting personal data: {str(e)}")
            return {}
        
    def _calculate_confidence(self, text: str, data_type: str) -> float:
        """
        calculate confidence score for a detected data type in the text.
        """
        try:
            if not text or data_type not in self.patterns:
                return 0.0
            matches = self.patterns[data_type].findall(text)
            if not matches:
                return 0.0
            base_confidence = min(len(matches) / 2, 1.0) * 0.6
            context_words = {
                'email': ['contact', 'email', '@', 'mail', 'address', 'e-mail', 'Email', 'EMAIL'],
                'phone': ['call', 'phone', 'tel', 'contact', 'mobile', 'number', 'Phone', 'PHONE'],
                'ssn': ['ssn', 'social', 'security', 'number', 'identification', 'SSN'],
                'credit_card': ['card', 'credit', 'payment', 'visa', 'mastercard', 'amex'],
                'date_of_birth': ['birth', 'born', 'dob', 'birthday', 'date', 'DOB'],
                'ip_address': ['ip', 'address', 'server', 'network', 'host', 'IP'],
                'url': ['website', 'link', 'site', 'web', 'http', 'https', 'URL']
            }
            context_score = 0.0
            if data_type in context_words:
                text_lower = text.lower()
                context_matches = sum(1 for word in context_words[data_type] if word.lower() in text_lower)
                context_score = min(context_matches / len(context_words[data_type]), 1.0) * 0.4
            format_score = 0.3 if self._validate_format(matches[0], data_type) else 0.0
            field_boost = 0.2 if any(
                field.lower() == data_type.replace('_', ' ') or
                field.lower() == data_type or
                field.lower() in [w.lower() for w in context_words.get(data_type, [])]
                for field in text.split()
            ) else 0.0
            confidence = min(
                base_confidence + context_score + format_score + field_boost,
                1.0
            )
            self.logger.debug(
                f"confidence for {data_type}: base={base_confidence:.2f}, "
                f"context={context_score:.2f}, format={format_score:.2f}, "
                f"field_boost={field_boost:.2f}, total={confidence:.2f}"
            )
            return confidence
        except Exception as e:
            self.logger.error(f"error calculating confidence: {str(e)}")
            return 0.0
        
    def _validate_format(self, value: str, data_type: str) -> bool:
        """
        validate the format of a detected value for a given data type.
        """
        try:
            if data_type == 'email':
                return '@' in value and '.' in value.split('@')[1]
            elif data_type == 'phone':
                digits = ''.join(filter(str.isdigit, value))
                return 10 <= len(digits) <= 15
            elif data_type == 'credit_card':
                digits = ''.join(filter(str.isdigit, value))
                if len(digits) < 13 or len(digits) > 19:
                    return False
                total = 0
                is_even = False
                for d in reversed(digits):
                    d = int(d)
                    if is_even:
                        d *= 2
                        if d > 9:
                            d -= 9
                    total += d
                    is_even = not is_even
                return total % 10 == 0
            elif data_type == 'ip_address':
                parts = value.split('.')
                return len(parts) == 4 and all(0 <= int(p) <= 255 for p in parts)
            return True
        except Exception:
            return False
    
    def discover_profile_data(self, query: str) -> List[Dict]:
        """
        search for profiles matching the query and extract personal data from them.
        returns a list of dictionaries with extracted data.
        """
        if not query or not self.ecosystem:
            self.logger.debug("empty query or no ecosystem provided")
            return []
        try:
            profiles = self.ecosystem.search_profiles(query)
            self.logger.debug(f"found {len(profiles)} profiles")
            if not profiles:
                self.logger.info(f"no profiles found for query: {query}")
                return []
            results = []
            for profile in profiles:
                self.logger.debug(f"processing profile: {profile}")
                if isinstance(profile, dict):
                    content = f"Email: {profile.get('email', '')}\n"
                    content += f"Phone: {profile.get('phone', '')}\n"
                    for key, value in profile.items():
                        if key not in ['id', 'email', 'phone', 'visibility']:
                            content += f"{key}: {value}\n"
                else:
                    content = getattr(profile, 'content', str(profile))
                self.logger.debug(f"profile content: {content}")
                personal_data = self._extract_personal_data(content)
                self.logger.debug(f"extracted personal data: {personal_data}")
                if not personal_data:
                    self.logger.debug("no personal data found in profile")
                    continue
                confidence_scores = {
                    data_type: self._calculate_confidence(content, data_type)
                    for data_type in personal_data.keys()
                }
                self.logger.debug(f"confidence scores: {confidence_scores}")
                filtered_data = {
                    data_type: values
                    for data_type, values in personal_data.items()
                    if confidence_scores[data_type] >= self.config.confidence_threshold
                }
                self.logger.debug(f"filtered data: {filtered_data}")
                if filtered_data:
                    profile_id = profile['id'] if isinstance(profile, dict) else getattr(profile, 'id', None)
                    profile_metadata = {
                        'visibility': profile.get('visibility') if isinstance(profile, dict) else getattr(profile, 'visibility', None)
                    }
                    result = {
                        'profile_id': profile_id,
                        'personal_data': filtered_data,
                        'confidence_scores': confidence_scores,
                        'profile_metadata': profile_metadata
                    }
                    self.logger.debug(f"adding result: {result}")
                    results.append(result)
            self.logger.info(f"found {len(results)} profiles with personal data")
            return results
        except Exception as e:
            self.logger.error(f"error discovering profile data: {str(e)}")
            return []
    
    def _retry_operation(self, operation):
        """
        try an operation a few times
        """
        import time
        retries = 0
        last_error = None
        while retries < self.config.max_retries:
            try:
                return operation()
            except Exception as e:
                last_error = e
                retries += 1
                if retries == self.config.max_retries:
                    self.logger.error(f"operation failed after {retries} retries: {str(e)}")
                    raise last_error
                wait_time = self.config.retry_delay * (2 ** (retries - 1))
                self.logger.warning(f"operation failed (attempt {retries}), retrying in {wait_time}s...")
                time.sleep(wait_time) 