"""
digital ecosystem simulation for the personal data assistant.
this module simulates user profiles and posts for agent evaluation.
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
    """
    configuration for the digital ecosystem simulation.
    """
    data_path: str = 'experiments/data'
    access_delay: Tuple[float, float] = (0.1, 0.5)
    error_rate: float = 0.1
    cache_size: int = 1000
    rate_limit: int = 100
    rate_window: int = 3600

class DigitalEcosystem:
    """
    class for simulating a digital environment with user profiles and posts.
    provides methods for searching and retrieving synthetic data.
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
        """
        load profiles and posts from data files.
        raises an error if files are missing.
        """
        try:
            with open(os.path.join(self.config.data_path, 'profiles.json'), 'r') as f:
                profiles = json.load(f)
                self._profiles = {p['id']: p for p in profiles}
            with open(os.path.join(self.config.data_path, 'posts.json'), 'r') as f:
                posts = json.load(f)
                self._posts = {p['id']: p for p in posts}
        except FileNotFoundError:
            raise RuntimeError("data files not found. run data_generator.py first.")

    def _initialize_test_profiles(self) -> List[Dict[str, Any]]:
        """
        create a set of test profiles with a mix of real, noisy, and obfuscated data.
        """
        names = ['John Doe', 'Jane Smith', 'Bob Wilson', 'Alice Brown', 'Eve Adams', 'Tom Lee', 'Sara King', 'Mike Fox', 'Nina Ray', 'Omar Zed']
        email_bases = ['john.doe', 'jane.smith', 'bob.wilson', 'alice.brown', 'eve.adams', 'tom.lee', 'sara.king', 'mike.fox', 'nina.ray', 'omar.zed']
        domains = ['example.com', 'email.com', 'web.net', 'site.org', 'mail.co']
        visibilities = ['public', 'private']
        profiles = []
        for i in range(50):
            idx = random.randint(0, 9)
            name = names[idx]
            if random.random() > 0.2:
                email = f"{email_bases[idx]}@{random.choice(domains)}"
                if random.random() < 0.2:
                    email = email.replace('@', ' at ')
            else:
                email = ''
            if random.random() > 0.3:
                phone = f"{random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
                if random.random() < 0.2:
                    phone = phone.replace('-', ' ')
            else:
                phone = ''
            ssn = f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}" if random.random() < 0.3 else ''
            credit_card = f"{random.randint(4000,5999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}" if random.random() < 0.2 else ''
            nickname = name.split()[0].lower() + str(random.randint(1,99)) if random.random() < 0.5 else ''
            address = f"{random.randint(100,999)} Main St" if random.random() < 0.3 else ''
            visibility = random.choice(visibilities)
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
            gt = []
            for field in ['email', 'phone', 'ssn', 'credit_card']:
                if profile.get(field):
                    gt.append(field)
            profile['ground_truth'] = gt
            profiles.append(profile)
        return profiles

    def _check_rate_limit(self) -> bool:
        """
        check if the rate limit for requests has been exceeded.
        """
        now = time.time()
        self.request_times = [t for t in self.request_times if now - t < self.config.rate_window]
        if len(self.request_times) >= self.config.rate_limit:
            return False
        self.request_times.append(now)
        return True

    def _simulate_delay(self):
        """
        simulate network or server delay.
        """
        delay = random.uniform(self.config.access_delay[0], self.config.access_delay[1])
        time.sleep(delay)

    def _simulate_error(self) -> bool:
        """
        randomly simulate an error based on the configured error rate.
        """
        return random.random() < self.config.error_rate

    def search_profiles(self, query: str) -> List[Dict[str, Any]]:
        """
        search for profiles matching the query string.
        may raise errors or be slow depending on simulation settings.
        """
        if not self._check_rate_limit():
            raise Exception("rate limit exceeded")
        self._simulate_delay()
        if query and self._simulate_error():
            raise Exception("search operation failed")
        query = query.lower()
        results = []
        for profile in self.profiles:
            if any(str(value).lower().find(query) != -1 for value in profile.values()):
                results.append(profile.copy())
        return results

    def get_profile(self, profile_id: str) -> Dict[str, Any]:
        """
        fetch a profile by id.
        """
        if not self._check_rate_limit():
            raise Exception("rate limit exceeded")
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("profile lookup failed")
        for profile in self.profiles:
            if profile['id'] == profile_id:
                return profile.copy()
        return None

    def get_profile_posts(self, profile_id: str) -> List[Dict[str, Any]]:
        """
        get all posts for a profile.
        """
        if not self._check_rate_limit():
            raise Exception("rate limit exceeded")
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("failed to retrieve posts")
        return [post for post in self._posts.values() if post['profile_id'] == profile_id]

    def get_post_by_id(self, post_id: str) -> Optional[Dict[str, Any]]:
        """
        fetch a post by id.
        """
        if not self._check_rate_limit():
            raise Exception("rate limit exceeded")
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("failed to retrieve post")
        return self._posts.get(post_id)

    def get_profile_by_id(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """
        fetch a profile by id (alternative method).
        """
        if not self._check_rate_limit():
            raise Exception("rate limit exceeded")
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("failed to retrieve profile")
        for profile in self.profiles:
            if profile['id'] == profile_id:
                return profile.copy()
        return None

    def search_posts(self, query: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        search posts for a keyword, optionally within a date range.
        """
        if not self._check_rate_limit():
            raise Exception("rate limit exceeded")
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("failed to search posts")
        query = query.lower()
        results = []
        for post in self._posts.values():
            if query in post.get('content', '').lower():
                if start_date and post['timestamp'] < start_date:
                    continue
                if end_date and post['timestamp'] > end_date:
                    continue
                results.append(post.copy())
        return results

    def request_data_removal(self, data_id: str, removal_type: str) -> Dict[str, Any]:
        """simulate a data removal request. doesn't actually delete anything, but pretends to"""
        if not self._check_rate_limit():
            raise Exception("rate limit exceeded")
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("removal request failed")
        return {
            'request_id': f"removal-{data_id}-{random.randint(1000,9999)}",
            'status': 'pending',
            'type': removal_type,
            'timestamp': datetime.now().isoformat()
        }

    def get_removal_request_status(self, request_id: str) -> Dict[str, Any]:
        """check the status of a removal request. always returns 'pending' for now"""
        if not self._check_rate_limit():
            raise Exception("rate limit exceeded")
        self._simulate_delay()
        if self._simulate_error():
            raise Exception("failed to check removal status")
        return {
            'request_id': request_id,
            'status': 'pending',
            'timestamp': datetime.now().isoformat()
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