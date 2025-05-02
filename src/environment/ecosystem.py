"""
Digital Ecosystem Simulation for Personal Data Discovery & Removal Assistant.
This module provides a simulated environment where agents can discover and analyze personal data.
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import random
import numpy as np
from datetime import datetime, timedelta
import time

@dataclass
class EcosystemConfig:
    """Configuration for the digital ecosystem"""
    data_path: str = 'experiments/data'
    access_delay: Tuple[float, float] = (0.1, 0.5)  # Random delay range in seconds
    error_rate: float = 0.1
    cache_size: int = 1000  # Number of items to cache
    rate_limit: int = 100
    rate_window: int = 3600  # 1 hour in seconds

class DigitalEcosystem:
    """
    Simulates a digital ecosystem for testing agent interactions.
    Provides controlled environment with simulated errors and delays.
    """
    
    def __init__(self, config: EcosystemConfig):
        self.config = config
        self._profiles: Dict[str, Dict] = {}
        self._posts: Dict[str, Dict] = {}
        self._cache: Dict[str, Any] = {}
        self.request_times = []
        self._load_data()
        self.profiles = self._initialize_test_profiles()

    def _load_data(self) -> None:
        """Load profiles and posts from data files"""
        try:
            with open(os.path.join(self.config.data_path, 'profiles.json'), 'r') as f:
                profiles = json.load(f)
                self._profiles = {p['id']: p for p in profiles}
            
            with open(os.path.join(self.config.data_path, 'posts.json'), 'r') as f:
                posts = json.load(f)
                self._posts = {p['id']: p for p in posts}
        except FileNotFoundError:
            raise RuntimeError("Data files not found. Run data_generator.py first.")

    def _initialize_test_profiles(self) -> List[Dict[str, Any]]:
        """Generate a diverse set of 50 test profiles with varied, noisy, and obfuscated data."""
        names = ['John Doe', 'Jane Smith', 'Bob Wilson', 'Alice Brown', 'Eve Adams', 'Tom Lee', 'Sara King', 'Mike Fox', 'Nina Ray', 'Omar Zed']
        email_bases = ['john.doe', 'jane.smith', 'bob.wilson', 'alice.brown', 'eve.adams', 'tom.lee', 'sara.king', 'mike.fox', 'nina.ray', 'omar.zed']
        domains = ['example.com', 'email.com', 'web.net', 'site.org', 'mail.co']
        visibilities = ['public', 'private']
        profiles = []
        for i in range(50):
            idx = random.randint(0, 9)
            name = names[idx]
            # Email with 80% chance, sometimes obfuscated
            if random.random() > 0.2:
                email = f"{email_bases[idx]}@{random.choice(domains)}"
                if random.random() < 0.2:
                    email = email.replace('@', ' at ')
            else:
                email = ''
            # Phone with 70% chance, sometimes obfuscated
            if random.random() > 0.3:
                phone = f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
                if random.random() < 0.2:
                    phone = phone.replace('-', ' ')
            else:
                phone = ''
            # SSN with 30% chance
            ssn = f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}" if random.random() < 0.3 else ''
            # Credit card with 20% chance
            credit_card = f"{random.randint(4000,5999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}" if random.random() < 0.2 else ''
            # Add some distractor fields
            nickname = name.split()[0].lower() + str(random.randint(1,99)) if random.random() < 0.5 else ''
            address = f"{random.randint(100,999)} Main St" if random.random() < 0.3 else ''
            # Visibility
            visibility = random.choice(visibilities)
            # Compose profile
            profile = {
                'id': str(i+1),
                'name': name,
                'email': email,
                'phone': phone,
                'ssn': ssn,
                'credit_card': credit_card,
                'nickname': nickname,
                'address': address,
                'visibility': visibility,
            }
            # Ground truth
            gt = []
            for field in ['email', 'phone', 'ssn', 'credit_card']:
                if profile.get(field):
                    gt.append(field)
            profile['ground_truth'] = gt
            profiles.append(profile)
        return profiles

    def _check_rate_limit(self) -> bool:
        """Check if current request is within rate limits"""
        now = time.time()
        
        # Remove old requests outside the window
        self.request_times = [t for t in self.request_times 
                            if now - t < self.config.rate_window]
        
        # Check if we're within limits
        if len(self.request_times) >= self.config.rate_limit:
            return False
            
        self.request_times.append(now)
        return True

    def _simulate_delay(self):
        """Simulate network/processing delay"""
        delay = random.uniform(self.config.access_delay[0], self.config.access_delay[1])
        time.sleep(delay)

    def _simulate_error(self) -> bool:
        """Simulate random errors"""
        return random.random() < self.config.error_rate

    def search_profiles(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for profiles matching the query string.
        Simulates basic search functionality with potential errors and delays.
        """
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
            
        self._simulate_delay()
        
        # In test environment, don't simulate errors for empty queries
        if query and self._simulate_error():
            raise Exception("Search operation failed")
            
        # Simple case-insensitive search
        query = query.lower()
        results = []
        
        for profile in self.profiles:
            # Match any field containing the query
            if any(str(value).lower().find(query) != -1 
                  for value in profile.values()):
                results.append(profile.copy())  # Return a copy to prevent modification
                
        return results

    def get_profile(self, profile_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific profile by ID.
        Simulates profile lookup with potential errors and delays.
        """
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
            
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("Profile lookup failed")
            
        for profile in self.profiles:
            if profile['id'] == profile_id:
                return profile.copy()
                
        return None

    def get_profile_posts(self, profile_id: str) -> List[Dict[str, Any]]:
        """Get all posts associated with a profile"""
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("Failed to retrieve posts")

        return [post for post in self._posts.values() 
                if post['profile_id'] == profile_id]

    def get_post_by_id(self, post_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific post by ID"""
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("Failed to retrieve post")

        return self._posts.get(post_id)

    def get_profile_by_id(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific profile by ID"""
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("Failed to retrieve profile")

        return self._profiles.get(profile_id)

    def search_posts(self, query: str, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search posts containing the query string with optional date range filtering
        """
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("Search operation failed")

        query = query.lower()
        results = []

        for post in self._posts.values():
            if query in post['content'].lower():
                if start_date and post['timestamp'] < start_date:
                    continue
                if end_date and post['timestamp'] > end_date:
                    continue
                results.append(post)

        return results

    def request_data_removal(self, data_id: str, 
                           removal_type: str) -> Dict[str, Any]:
        """
        Simulate a request to remove specific data
        Returns status of the removal request
        """
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("Removal request failed")

        # Simulate various removal outcomes
        success_rate = 0.8  # 80% success rate
        if random.random() < success_rate:
            status = "approved"
            processing_time = random.randint(1, 7)  # Days to process
        else:
            status = random.choice(["denied", "pending_verification"])
            processing_time = random.randint(7, 30)

        return {
            'request_id': f"rem_{random.randint(10000, 99999)}",
            'data_id': data_id,
            'status': status,
            'estimated_processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        }

    def get_removal_request_status(self, request_id: str) -> Dict[str, Any]:
        """Get the current status of a removal request"""
        if not self._check_rate_limit():
            raise Exception("Rate limit exceeded")
        
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("Failed to retrieve status")

        # Simulate status check
        return {
            'request_id': request_id,
            'current_status': random.choice([
                "in_progress", "completed", "verification_needed", "failed"
            ]),
            'last_updated': datetime.now().isoformat()
        }

def main():
    """Main function to test the ecosystem"""
    config = EcosystemConfig()
    ecosystem = DigitalEcosystem(config)
    
    # Test basic functionality
    try:
        profiles = ecosystem.search_profiles("john")
        if profiles:
            profile = profiles[0]
            print(f"Found profile: {profile['name']}")
            
            posts = ecosystem.get_profile_posts(profile['id'])
            print(f"Number of posts: {len(posts)}")
            
            if posts:
                removal_request = ecosystem.request_data_removal(
                    posts[0]['id'], "post_deletion"
                )
                print(f"Removal request status: {removal_request['status']}")
    
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == '__main__':
    main() 