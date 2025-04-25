import random
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime

class DigitalEnvironment:
    """Simulates a digital environment containing user profiles, posts, and articles."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the digital environment.
        
        Args:
            config: Configuration dictionary containing:
                - num_profiles: Number of user profiles to generate
                - noise_level: Level of noise in the data (0-1)
                - sensitive_data_ratio: Ratio of sensitive to non-sensitive data (0-1)
        """
        self.config = config
        self.profiles = []
        self.posts = []
        self.articles = []
        self.state = {
            'timestamp': None,
            'profiles': [],
            'posts': [],
            'articles': [],
            'noise_level': config.get('noise_level', 0.1)
        }
        self.initialize_environment()
    
    def initialize_environment(self):
        """Initialize the environment with synthetic data."""
        self._generate_profiles()
        self._generate_posts()
        self._generate_articles()
        self.update_state()
    
    def _generate_profiles(self):
        """Generate synthetic user profiles."""
        num_profiles = self.config.get('num_profiles', 100)
        sensitive_ratio = self.config.get('sensitive_data_ratio', 0.3)
        
        for i in range(num_profiles):
            profile = {
                'id': f'user_{i}',
                'name': f'User {i}',
                'email': f'user{i}@example.com',
                'phone': f'+1-555-{random.randint(1000, 9999)}',
                'address': f'{random.randint(1, 999)} Example St, City',
                'sensitive': random.random() < sensitive_ratio,
                'creation_date': datetime.now().isoformat()
            }
            self.profiles.append(profile)
    
    def _generate_posts(self):
        """Generate synthetic posts for users."""
        posts_per_user = random.randint(5, 15)
        sensitive_ratio = self.config.get('sensitive_data_ratio', 0.3)
        
        for profile in self.profiles:
            for _ in range(posts_per_user):
                post = {
                    'id': f'post_{len(self.posts)}',
                    'user_id': profile['id'],
                    'content': self._generate_post_content(profile['sensitive']),
                    'sensitive': random.random() < sensitive_ratio,
                    'timestamp': datetime.now().isoformat()
                }
                self.posts.append(post)
    
    def _generate_articles(self):
        """Generate synthetic articles."""
        num_articles = random.randint(50, 200)
        sensitive_ratio = self.config.get('sensitive_data_ratio', 0.3)
        
        for i in range(num_articles):
            article = {
                'id': f'article_{i}',
                'title': f'Article {i}',
                'content': self._generate_article_content(),
                'sensitive': random.random() < sensitive_ratio,
                'publication_date': datetime.now().isoformat()
            }
            self.articles.append(article)
    
    def _generate_post_content(self, is_sensitive: bool) -> str:
        """Generate synthetic post content.
        
        Args:
            is_sensitive: Whether the post should contain sensitive information
            
        Returns:
            Generated post content
        """
        templates = [
            "Just had a great day at {}!",
            "Can't wait to visit {} next week!",
            "Looking for recommendations in {}.",
            "Anyone want to meet up at {}?"
        ]
        
        locations = [
            "the park",
            "the mall",
            "downtown",
            "the beach"
        ]
        
        sensitive_info = [
            "my social security number is XXX-XX-XXXX",
            "my credit card number is XXXX-XXXX-XXXX-XXXX",
            "my password is ********",
            "my bank account number is XXXXXXXXXX"
        ]
        
        template = random.choice(templates)
        location = random.choice(locations)
        content = template.format(location)
        
        if is_sensitive:
            content += f" BTW, {random.choice(sensitive_info)}"
        
        return content
    
    def _generate_article_content(self) -> str:
        """Generate synthetic article content.
        
        Returns:
            Generated article content
        """
        paragraphs = random.randint(3, 7)
        content = []
        
        for _ in range(paragraphs):
            words = random.randint(50, 100)
            paragraph = " ".join([f"word_{i}" for i in range(words)])
            content.append(paragraph)
        
        return "\n\n".join(content)
    
    def update_state(self):
        """Update the environment state."""
        self.state['timestamp'] = datetime.now().isoformat()
        self.state['profiles'] = self.profiles
        self.state['posts'] = self.posts
        self.state['articles'] = self.articles
    
    def add_noise(self, data: str) -> str:
        """Add noise to data based on noise level.
        
        Args:
            data: Input data string
            
        Returns:
            Data with added noise
        """
        if random.random() > self.state['noise_level']:
            return data
            
        noise_types = [
            lambda x: x.replace('a', '@'),
            lambda x: x.replace('e', '3'),
            lambda x: x.replace('i', '1'),
            lambda x: x.replace('o', '0'),
            lambda x: x.replace('s', '$')
        ]
        
        noisy_data = data
        for _ in range(random.randint(1, 3)):
            noise_func = random.choice(noise_types)
            noisy_data = noise_func(noisy_data)
        
        return noisy_data
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the environment.
        
        Returns:
            Current environment state
        """
        return self.state
    
    def reset(self):
        """Reset the environment to initial state."""
        self.profiles = []
        self.posts = []
        self.articles = []
        self.initialize_environment() 