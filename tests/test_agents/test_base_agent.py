"""Tests for the base agent functionality"""

import pytest
from src.agents.base_agent import BaseAgent

class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing"""
    def perceive(self, environment):
        return {"test": "perception"}
    
    def decide(self, perception):
        return {"test": "decision"}
    
    def act(self, decision):
        return {"test": "action"}

class TestBaseAgent:
    def test_base_agent_initialization(self):
        """Test basic initialization of base agent"""
        agent = ConcreteAgent(agent_id="test_agent")
        assert agent is not None
        assert hasattr(agent, 'agent_id')
        assert agent.agent_id == "test_agent"
        assert hasattr(agent, 'config')
        assert hasattr(agent, 'state')
        assert hasattr(agent, 'performance_metrics')

    def test_base_agent_logging(self):
        """Test that base agent can log messages"""
        agent = ConcreteAgent(agent_id="test_agent")
        # This should not raise any exceptions
        agent.logger.info("Test message")
        agent.logger.debug("Test debug message")
        agent.logger.warning("Test warning message")

    def test_config_initialization(self):
        """Test configuration initialization with different inputs"""
        # Test with empty config
        agent1 = ConcreteAgent(agent_id="test1")
        assert agent1.config == {}

        # Test with custom config
        custom_config = {"key1": "value1", "key2": 42}
        agent2 = ConcreteAgent(agent_id="test2", config=custom_config)
        assert agent2.config == custom_config

        # Test config immutability
        custom_config["key3"] = "value3"
        assert "key3" not in agent2.config

    def test_state_management(self):
        """Test state initialization and updates"""
        agent = ConcreteAgent(agent_id="test_agent")
        
        # Initial state should be empty
        assert agent.state == {}
        
        # Test state updates
        test_state = {"position": (0, 0), "status": "active"}
        agent.state = test_state
        assert agent.state == test_state
        
        # Test state is a copy
        test_state["new_key"] = "new_value"
        assert "new_key" not in agent.state

    def test_performance_metrics_initialization(self):
        """Test that performance metrics are properly initialized"""
        agent = ConcreteAgent(agent_id="test_agent")
        expected_metrics = {
            "processing_time": 0.0,
            "success_rate": 0.0,
            "error_rate": 0.0
        }
        assert agent.performance_metrics == expected_metrics

    def test_update_metrics(self):
        """Test updating performance metrics"""
        agent = ConcreteAgent(agent_id="test_agent")
        
        # Test updating single metric
        agent.update_metrics({"processing_time": 1.5})
        assert agent.performance_metrics["processing_time"] == 1.5
        assert agent.performance_metrics["success_rate"] == 0.0
        
        # Test updating multiple metrics
        agent.update_metrics({
            "success_rate": 0.8,
            "error_rate": 0.2
        })
        assert agent.performance_metrics["success_rate"] == 0.8
        assert agent.performance_metrics["error_rate"] == 0.2

    def test_get_metrics(self):
        """Test retrieving performance metrics"""
        agent = ConcreteAgent(agent_id="test_agent")
        agent.update_metrics({
            "processing_time": 2.0,
            "success_rate": 0.9,
            "error_rate": 0.1
        })
        metrics = agent.get_metrics()
        assert metrics == agent.performance_metrics
        
        # Test metrics are returned as copy
        metrics["processing_time"] = 5.0
        assert agent.performance_metrics["processing_time"] == 2.0

    def test_reset(self):
        """Test resetting agent state and metrics"""
        agent = ConcreteAgent(agent_id="test_agent")
        
        # Set some state and metrics
        agent.state = {"test": "value"}
        agent.update_metrics({
            "processing_time": 1.0,
            "success_rate": 0.5,
            "error_rate": 0.5
        })
        
        # Reset agent
        agent.reset()
        
        # Verify state and metrics are reset
        assert agent.state == {}
        assert agent.performance_metrics["processing_time"] == 0.0
        assert agent.performance_metrics["success_rate"] == 0.0
        assert agent.performance_metrics["error_rate"] == 0.0

    def test_abstract_methods(self):
        """Test that abstract methods raise NotImplementedError"""
        # Create a new class that inherits from BaseAgent but doesn't implement abstract methods
        class IncompleteAgent(BaseAgent):
            pass
            
        # Verify that we can't instantiate BaseAgent directly
        with pytest.raises(TypeError) as excinfo:
            BaseAgent(agent_id="test_agent")
        assert "Can't instantiate abstract class BaseAgent" in str(excinfo.value)
        
        # Verify that we can't instantiate a class that doesn't implement all abstract methods
        with pytest.raises(TypeError) as excinfo:
            IncompleteAgent(agent_id="test_agent")
        assert "Can't instantiate abstract class IncompleteAgent" in str(excinfo.value)

    def test_concrete_implementation(self):
        """Test concrete implementation of BaseAgent"""
        agent = ConcreteAgent(agent_id="test_concrete")
        
        # Test perceive method
        perception = agent.perceive({})
        assert perception == {"test": "perception"}
        
        # Test decide method
        decision = agent.decide({})
        assert decision == {"test": "decision"}
        
        # Test act method
        action = agent.act({})
        assert action == {"test": "action"} 