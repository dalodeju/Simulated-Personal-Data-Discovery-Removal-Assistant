"""Personal data discovery using regex patterns.
Finds stuff like emails, phone numbers, SSNs, etc. in text.
"""

import re
import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.environment.ecosystem import DigitalEcosystem, EcosystemConfig
from src.utils.common import setup_logging, load_config
import logging

# Load configuration
config = load_config().get('web_scraper', {
    'max_retries': 3,
    'retry_delay': 1.0,
    'confidence_threshold': 0.7
})

# Regex patterns for finding personal data
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
    """Custom exception for input validation errors"""
    pass

@dataclass
class ScraperConfig:
    """Configuration for the WebScraperAgent"""
    max_retries: int = config.get('max_retries', 3)
    retry_delay: float = config.get('retry_delay', 1.0)
    confidence_threshold: float = config.get('confidence_threshold', 0.7)
    
    def __post_init__(self):
        """Validate configuration values"""
        if self.max_retries < 1:
            raise ValidationError("max_retries must be at least 1")
        if self.retry_delay < 0:
            raise ValidationError("retry_delay cannot be negative")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValidationError("confidence_threshold must be between 0 and 1")

class WebScraperAgent:
    """Agent for discovering personal data in text using pattern matching."""
    
    def __init__(self, ecosystem: Optional[DigitalEcosystem] = None, config: Optional[ScraperConfig] = None):
        """Initialize WebScraperAgent with ecosystem and config"""
        super().__init__()
        self.ecosystem = ecosystem
        self.config = config or ScraperConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize regex patterns for personal data discovery
        self.patterns = {
            'email': re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            'phone': re.compile(r'\+?1?\d{9,15}|\(\d{3}\)\s*\d{3}[-.]?\d{4}|\d{3}[-.]?\d{3}[-.]?\d{4}'),
            'ssn': re.compile(r'\d{3}-\d{2}-\d{4}'),
            'credit_card': re.compile(r'\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}'),
            'date_of_birth': re.compile(r'\d{2}[/-]\d{2}[/-]\d{4}'),
            'ip_address': re.compile(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'),
            'url': re.compile(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*')
        }
        
        # Setup logging
        self.logger.info("Initializing web scraper agent...")
        
    def _extract_personal_data(self, text: str) -> Dict[str, List[str]]:
        """
        Extract personal data from text using regex patterns.
        
        Args:
            text: Text to scan for personal data
            
        Returns:
            Dictionary mapping data types to lists of found values
        """
        # Handle None or empty input gracefully
        if text is None:
            self.logger.warning("None value provided for scanning")
            return {}
            
        if not isinstance(text, str):
            text = str(text)
            self.logger.warning(f"Converting non-string input to string: {type(text)}")
            
        if not text.strip():
            self.logger.warning("Empty text provided for scanning")
            return {}
            
        results = {}
        
        try:
            # Look for each type of data
            for data_type, pattern in self.patterns.items():
                # Find all matches
                matches = pattern.findall(text)
                if matches:
                    # Clean and format matches
                    cleaned_matches = []
                    for match in matches:
                        if isinstance(match, tuple):
                            # Handle grouped matches (like phone numbers)
                            cleaned_match = '-'.join(match)
                        else:
                            cleaned_match = match.strip()
                        if cleaned_match:
                            cleaned_matches.append(cleaned_match)
                            
                    if cleaned_matches:
                        self.logger.debug(f"Found {len(cleaned_matches)} matches for {data_type}")
                        results[data_type] = cleaned_matches
                    
            return results
        except Exception as e:
            self.logger.error(f"Error extracting personal data: {str(e)}")
            return {}
            
    def _calculate_confidence(self, text: str, data_type: str) -> float:
        """
        Calculate confidence score for a particular type of data.
        
        Args:
            text: Text containing the data
            data_type: Type of personal data to calculate confidence for
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            if not text or data_type not in self.patterns:
                return 0.0
                
            matches = self.patterns[data_type].findall(text)
            if not matches:
                return 0.0
                
            # Base confidence from matches (up to 0.6)
            base_confidence = min(len(matches) / 2, 1.0) * 0.6
            
            # Context words that increase confidence
            context_words = {
                'email': ['contact', 'email', '@', 'mail', 'address', 'e-mail', 'Email', 'EMAIL'],
                'phone': ['call', 'phone', 'tel', 'contact', 'mobile', 'number', 'Phone', 'PHONE'],
                'ssn': ['ssn', 'social', 'security', 'number', 'identification', 'SSN'],
                'credit_card': ['card', 'credit', 'payment', 'visa', 'mastercard', 'amex'],
                'date_of_birth': ['birth', 'born', 'dob', 'birthday', 'date', 'DOB'],
                'ip_address': ['ip', 'address', 'server', 'network', 'host', 'IP'],
                'url': ['website', 'link', 'site', 'web', 'http', 'https', 'URL']
            }
            
            # Context score (up to 0.4)
            context_score = 0.0
            if data_type in context_words:
                # Check for context words in the text
                text_lower = text.lower()
                context_matches = sum(1 for word in context_words[data_type] if word.lower() in text_lower)
                context_score = min(context_matches / len(context_words[data_type]), 1.0) * 0.4
                
            # Format validation (0.3 if valid)
            format_score = 0.3 if self._validate_format(matches[0], data_type) else 0.0
                
            # Additional boost for exact field matches (0.2)
            field_boost = 0.2 if any(
                field.lower() == data_type.replace('_', ' ') or
                field.lower() == data_type or
                field.lower() in [w.lower() for w in context_words.get(data_type, [])]
                for field in text.split()
            ) else 0.0
                
            # Combine scores with proper weighting
            confidence = min(
                base_confidence + context_score + format_score + field_boost,
                1.0
            )
            
            self.logger.debug(
                f"Confidence for {data_type}: base={base_confidence:.2f}, "
                f"context={context_score:.2f}, format={format_score:.2f}, "
                f"field_boost={field_boost:.2f}, total={confidence:.2f}"
            )
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.0
            
    def _validate_format(self, value: str, data_type: str) -> bool:
        """Additional format validation for specific data types"""
        try:
            if data_type == 'email':
                # Check for valid email format
                return '@' in value and '.' in value.split('@')[1]
                
            elif data_type == 'phone':
                # Check for valid phone number length
                digits = ''.join(filter(str.isdigit, value))
                return 10 <= len(digits) <= 15
                
            elif data_type == 'credit_card':
                # Basic Luhn algorithm check
                digits = ''.join(filter(str.isdigit, value))
                if len(digits) < 13 or len(digits) > 19:
                    return False
                    
                # Luhn algorithm
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
                # Check valid IP address format
                parts = value.split('.')
                return len(parts) == 4 and all(0 <= int(p) <= 255 for p in parts)
                
            return True  # Default to True for other types
            
        except Exception:
            return False
        
    def discover_profile_data(self, query: str) -> List[Dict]:
        """
        Discover and extract personal data from profiles matching the query.
        
        Args:
            query: Search query to find relevant profiles
            
        Returns:
            List of dictionaries containing extracted profile data and confidence scores
        """
        if not query or not self.ecosystem:
            self.logger.debug("Empty query or no ecosystem")
            return []
            
        try:
            # Search for matching profiles
            profiles = self.ecosystem.search_profiles(query)
            self.logger.debug(f"Found {len(profiles)} profiles")
            
            if not profiles:
                self.logger.info(f"No profiles found matching query: {query}")
                return []
                
            results = []
            for profile in profiles:
                self.logger.debug(f"Processing profile: {profile}")
                
                # Convert profile to string for scanning if it's a dictionary
                if isinstance(profile, dict):
                    # Create a string representation of the profile content
                    content = f"Email: {profile.get('email', '')}\n"
                    content += f"Phone: {profile.get('phone', '')}\n"
                    # Add any other fields that might contain personal data
                    for key, value in profile.items():
                        if key not in ['id', 'email', 'phone', 'visibility']:
                            content += f"{key}: {value}\n"
                else:
                    content = getattr(profile, 'content', str(profile))
                
                self.logger.debug(f"Profile content: {content}")
                
                # Extract personal data from profile content
                personal_data = self._extract_personal_data(content)
                self.logger.debug(f"Extracted personal data: {personal_data}")
                
                if not personal_data:
                    self.logger.debug("No personal data found in profile")
                    continue
                    
                # Calculate confidence scores for each data type
                confidence_scores = {
                    data_type: self._calculate_confidence(content, data_type)
                    for data_type in personal_data.keys()
                }
                self.logger.debug(f"Confidence scores: {confidence_scores}")
                
                # Filter out low confidence results
                filtered_data = {
                    data_type: values
                    for data_type, values in personal_data.items()
                    if confidence_scores[data_type] >= self.config.confidence_threshold
                }
                self.logger.debug(f"Filtered data: {filtered_data}")
                
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
                    self.logger.debug(f"Adding result: {result}")
                    results.append(result)
                    
            self.logger.info(f"Found {len(results)} profiles with personal data")
            return results
            
        except Exception as e:
            self.logger.error(f"Error discovering profile data: {str(e)}")
            return []
            
    def _retry_operation(self, operation):
        """
        Retry an operation with exponential backoff.
        
        Args:
            operation: Function to retry
            
        Returns:
            Result of the operation
            
        Raises:
            Exception: If all retries fail
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
                    self.logger.error(f"Operation failed after {retries} retries: {str(e)}")
                    raise last_error
                    
                wait_time = self.config.retry_delay * (2 ** (retries - 1))
                self.logger.warning(f"Operation failed (attempt {retries}), retrying in {wait_time}s...")
                time.sleep(wait_time) 