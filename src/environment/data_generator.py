"""
Data Generator for creating fake personal data in our simulated environment.
"""

import random
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import faker
import numpy as np
from dataclasses import dataclass

@dataclass
class DataConfig:
    """config for data generation"""
    num_profiles: int = 100
    num_posts_per_profile: tuple = (5, 20)  # (min, max)
    noise_level: float = 0.1  # 0.0 to 1.0
    sensitive_data_ratio: float = 0.3  # ratio of content containing sensitive data

class DataGenerator:
    """generates fake personal data for the simulated environment"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.fake = faker.Faker()
        self.sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{16}\b',  # credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # email
            r'\b\d{10}\b',  # phone number
        ]

    def generate_profile(self) -> Dict[str, Any]:
        """Generate a single user profile with synthetic personal data"""
        profile = {
            'id': self.fake.uuid4(),
            'username': self.fake.user_name(),
            'email': self.fake.email(),
            'name': self.fake.name(),
            'address': self.fake.address(),
            'phone': self.fake.phone_number(),
            'birth_date': self.fake.date_of_birth().isoformat(),
            'registration_date': self.fake.date_this_decade().isoformat(),
            'bio': self.fake.text(max_nb_chars=200),
            'occupation': self.fake.job(),
            'social_links': {
                'facebook': f"facebook.com/{self.fake.user_name()}",
                'twitter': f"twitter.com/{self.fake.user_name()}",
                'linkedin': f"linkedin.com/in/{self.fake.user_name()}"
            }
        }
        return profile

    def generate_post(self, profile_id: str, is_sensitive: bool = False) -> Dict[str, Any]:
        """generate a post, sometimes containing sensitive information"""
        post = {
            'id': self.fake.uuid4(),
            'profile_id': profile_id,
            'timestamp': self.fake.date_time_this_year().isoformat(),
            'content': self._generate_content(is_sensitive),
            'location': self.fake.city() if random.random() < 0.3 else None,
            'tags': self._generate_tags(),
            'visibility': random.choice(['public', 'private', 'friends']),
            'contains_sensitive_data': is_sensitive
        }
        return post

    def _generate_content(self, is_sensitive: bool) -> str:
        """Generate post content, optionally including sensitive information"""
        base_content = self.fake.text(max_nb_chars=280)
        if is_sensitive:
            sensitive_info = [
                f"SSN: {self.fake.ssn()}",
                f"Credit Card: {self.fake.credit_card_number()}",
                f"Phone: {self.fake.phone_number()}",
                f"Address: {self.fake.address()}"
            ]
            return f"{base_content}\n{random.choice(sensitive_info)}"
        return base_content

    def _generate_tags(self) -> List[str]:
        """Generate random tags for a post"""
        num_tags = random.randint(0, 5)
        return [self.fake.word() for _ in range(num_tags)]

    def _add_noise(self, text: str) -> str:
        """Add random noise to text based on noise level"""
        if random.random() < self.config.noise_level:
            # Add typos, extra spaces, or special characters
            modifications = [
                lambda t: t.replace(' ', '  '),
                lambda t: t.replace('a', '@'),
                lambda t: t.replace('i', '1'),
                lambda t: t + ' ' + ''.join(random.choices('!@#$%^&*()', k=3))
            ]
            return random.choice(modifications)(text)
        return text

    def generate_dataset(self, output_dir: str = 'experiments/data') -> None:
        """Generate complete dataset with profiles and posts"""
        os.makedirs(output_dir, exist_ok=True)
        
        profiles = []
        posts = []
        
        # generate profiles
        for _ in range(self.config.num_profiles):
            profile = self.generate_profile()
            profiles.append(profile)
            
            # generate posts for each profile
            num_posts = random.randint(*self.config.num_posts_per_profile)
            for _ in range(num_posts):
                is_sensitive = random.random() < self.config.sensitive_data_ratio
                post = self.generate_post(profile['id'], is_sensitive)
                post['content'] = self._add_noise(post['content'])
                posts.append(post)

        # save to files
        with open(os.path.join(output_dir, 'profiles.json'), 'w') as f:
            json.dump(profiles, f, indent=2)
        
        with open(os.path.join(output_dir, 'posts.json'), 'w') as f:
            json.dump(posts, f, indent=2)

def main():
    """Main function to generate dataset"""
    config = DataConfig(
        num_profiles=100,
        num_posts_per_profile=(5, 20),
        noise_level=0.1,
        sensitive_data_ratio=0.3
    )
    
    generator = DataGenerator(config)
    generator.generate_dataset()

if __name__ == '__main__':
    main() 
