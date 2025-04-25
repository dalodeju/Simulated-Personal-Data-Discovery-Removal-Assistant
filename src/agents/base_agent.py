"""Base class for all agents in the system."""

from abc import ABC, abstractmethod
import logging
from typing import Any, Dict, List
from copy import deepcopy

class BaseAgent(ABC):
    """Base class for all agents in the system."""
    
    def __init__(self, agent_id: str, config: Dict[str, Any] = None):
        """Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            config: Configuration dictionary for the agent
        """
        self.agent_id = agent_id
        self.config = deepcopy(config) if config else {}
        self._state = {}  # Make state private
        self.performance_metrics = {
            'processing_time': 0.0,
            'success_rate': 0.0,
            'error_rate': 0.0
        }
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{self.agent_id}")
    
    @property
    def state(self) -> Dict[str, Any]:
        """Get the agent's state."""
        return deepcopy(self._state)
    
    @state.setter
    def state(self, value: Dict[str, Any]) -> None:
        """Set the agent's state."""
        self._state = deepcopy(value)
    
    @abstractmethod
    def perceive(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Process input from the environment.
        
        Args:
            environment_state: Current state of the environment
            
        Returns:
            Processed perception data
        """
        pass
    
    @abstractmethod
    def decide(self, perception_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make decisions based on perceived data.
        
        Args:
            perception_data: Processed perception data
            
        Returns:
            Decision data
        """
        pass
    
    @abstractmethod
    def act(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute decided actions.
        
        Args:
            decision: Decision data
            
        Returns:
            Results of the actions
        """
        pass
    
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update the agent's performance metrics.
        
        Args:
            metrics: Dictionary of metrics to update
        """
        for key, value in metrics.items():
            if key in self.performance_metrics:
                self.performance_metrics[key] = value
    
    def get_metrics(self) -> Dict[str, float]:
        """Get the agent's performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        return deepcopy(self.performance_metrics)
    
    def reset(self) -> None:
        """Reset the agent's state and metrics."""
        self._state = {}
        self.performance_metrics = {
            'processing_time': 0.0,
            'success_rate': 0.0,
            'error_rate': 0.0
        } 