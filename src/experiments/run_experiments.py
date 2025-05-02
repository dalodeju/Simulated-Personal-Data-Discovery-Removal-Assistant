"""
experiment runner for the personal data project.
this module runs experiments, applies different strategies, and collects results.
"""

import os
import json
import time
from typing import Dict, List, Any
import numpy as np
from datetime import datetime

from environment.simulator import DigitalEnvironment
from src.agents.web_scraper import WebScraperAgent
from src.agents.data_analyzer import DataAnalyzerAgent
from src.agents.risk_evaluator import RiskEvaluatorAgent
from src.agents.recommender import RecommenderAgent
from src.utils.metrics import PerformanceMetrics
from src.utils.visualization import ExperimentVisualizer

class ExperimentRunner:
    """
    class for running and managing experiments, including environment and agent setup.
    """
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.results_dir = os.path.join('experiments', 'results',
                                      datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.results_dir, exist_ok=True)
        self.environment = None
        self.agents = {}
        self.metrics = PerformanceMetrics()
        self.visualizer = ExperimentVisualizer()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        load experiment configuration from file or use defaults.
        """
        default_config = {
            'environment': {
                'num_profiles': 100,
                'noise_level': 0.1,
                'sensitive_data_ratio': 0.3
            },
            'agents': {
                'web_scraper': {
                    'search_depth': 3,
                    'batch_size': 10
                },
                'data_analyzer': {
                    'sensitivity_threshold': 0.7,
                    'nlp_model': 'en_core_web_sm'
                },
                'risk_evaluator': {
                    'risk_threshold': 0.7,
                    'feature_weights': {
                        'sensitivity_score': 0.4,
                        'exposure_level': 0.3,
                        'data_age': 0.2,
                        'source_reliability': 0.1
                    }
                },
                'recommender': {
                    'recommendation_threshold': 0.6,
                    'max_recommendations': 3
                }
            },
            'experiments': {
                'num_runs': 10,
                'noise_levels': [0.0, 0.1, 0.2, 0.3, 0.4],
                'coordination_strategies': ['sequential', 'parallel', 'hybrid']
            }
        }
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self._deep_update(default_config, user_config)
        return default_config
    
    def _deep_update(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """
        recursively update a dictionary with another dictionary.
        """
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    
    def setup_environment(self) -> None:
        """
        set up the digital environment for experiments.
        """
        self.environment = DigitalEnvironment(self.config['environment'])
    
    def setup_agents(self) -> None:
        """
        set up all agents with their respective configurations.
        """
        self.agents['web_scraper'] = WebScraperAgent('web_scraper_1',
                                                    self.config['agents']['web_scraper'])
        self.agents['data_analyzer'] = DataAnalyzerAgent('data_analyzer_1',
                                                        self.config['agents']['data_analyzer'])
        self.agents['risk_evaluator'] = RiskEvaluatorAgent('risk_evaluator_1',
                                                          self.config['agents']['risk_evaluator'])
        self.agents['recommender'] = RecommenderAgent('recommender_1',
                                                     self.config['agents']['recommender'])
    
    def run_experiment(self, coordination_strategy: str = 'sequential') -> Dict[str, Any]:
        """
        run a single experiment with the specified coordination strategy.
        returns a dictionary of results and metrics.
        """
        start_time = time.time()
        self.environment.reset()
        for agent in self.agents.values():
            agent.reset()
        env_state = self.environment.get_state()
        if coordination_strategy == 'sequential':
            results = self._run_sequential()
        elif coordination_strategy == 'parallel':
            results = self._run_parallel()
        else:
            results = self._run_hybrid()
        processing_time = time.time() - start_time
        agent_states = {
            name: agent.get_metrics()
            for name, agent in self.agents.items()
        }
        return {
            'coordination_strategy': coordination_strategy,
            'processing_time': processing_time,
            'agent_states': agent_states,
            'results': results
        }
    
    def _run_sequential(self) -> Dict[str, Any]:
        """
        run all agents sequentially in a pipeline.
        returns a dictionary of results.
        """
        env_state = self.environment.get_state()
        scraper_perception = self.agents['web_scraper'].perceive(env_state)
        scraper_decisions = self.agents['web_scraper'].decide(scraper_perception)
        scraper_results = self.agents['web_scraper'].act(scraper_decisions)
        analyzer_state = {**env_state, 'discovered_items': scraper_results['processed_items']}
        analyzer_perception = self.agents['data_analyzer'].perceive(analyzer_state)
        analyzer_decisions = self.agents['data_analyzer'].decide(analyzer_perception)
        analyzer_results = self.agents['data_analyzer'].act(analyzer_decisions)
        evaluator_state = {**env_state, 'analyzed_items': analyzer_results['flagged_items']}
        evaluator_perception = self.agents['risk_evaluator'].perceive(evaluator_state)
        evaluator_decisions = self.agents['risk_evaluator'].decide(evaluator_perception)
        evaluator_results = self.agents['risk_evaluator'].act(evaluator_decisions)
        recommender_state = {**env_state, 'flagged_items': evaluator_results['flagged_items']}
        recommender_perception = self.agents['recommender'].perceive(recommender_state)
        recommender_decisions = self.agents['recommender'].decide(recommender_perception)
        recommender_results = self.agents['recommender'].act(recommender_decisions)
        return {
            'scraper_results': scraper_results,
            'analyzer_results': analyzer_results,
            'evaluator_results': evaluator_results,
            'recommender_results': recommender_results
        }
    
    def _run_parallel(self) -> Dict[str, Any]:
        """
        run agents in parallel (simulated).
        returns a dictionary of results.
        """
        env_state = self.environment.get_state()
        perceptions = {
            'web_scraper': self.agents['web_scraper'].perceive(env_state),
            'data_analyzer': self.agents['data_analyzer'].perceive(env_state),
            'risk_evaluator': self.agents['risk_evaluator'].perceive(env_state),
            'recommender': self.agents['recommender'].perceive(env_state)
        }
        decisions = {
            name: agent.decide(perceptions[name])
            for name, agent in self.agents.items()
        }
        results = {
            name: agent.act(decisions[name])
            for name, agent in self.agents.items()
        }
        return results
    
    def _run_hybrid(self) -> Dict[str, Any]:
        """
        run agents in a hybrid sequential-parallel manner.
        returns a dictionary of results.
        """
        env_state = self.environment.get_state()
        scraper_perception = self.agents['web_scraper'].perceive(env_state)
        analyzer_perception = self.agents['data_analyzer'].perceive(env_state)
        scraper_decisions = self.agents['web_scraper'].decide(scraper_perception)
        analyzer_decisions = self.agents['data_analyzer'].decide(analyzer_perception)
        scraper_results = self.agents['web_scraper'].act(scraper_decisions)
        analyzer_results = self.agents['data_analyzer'].act(analyzer_decisions)
        evaluator_state = {**env_state, 'analyzed_items': analyzer_results['flagged_items']}
        recommender_state = {**env_state, 'flagged_items': []}
        evaluator_perception = self.agents['risk_evaluator'].perceive(evaluator_state)
        recommender_perception = self.agents['recommender'].perceive(recommender_state)
        evaluator_decisions = self.agents['risk_evaluator'].decide(evaluator_perception)
        recommender_decisions = self.agents['recommender'].decide(recommender_perception)
        evaluator_results = self.agents['risk_evaluator'].act(evaluator_decisions)
        recommender_results = self.agents['recommender'].act(recommender_decisions)
        return {
            'phase1': {
                'scraper_results': scraper_results,
                'analyzer_results': analyzer_results
            },
            'phase2': {
                'evaluator_results': evaluator_results,
                'recommender_results': recommender_results
            }
        }
    
    def run_all_experiments(self) -> None:
        """
        run all experiments according to the configuration.
        """
        self.setup_environment()
        self.setup_agents()
        all_results = []
        for noise_level in self.config['experiments']['noise_levels']:
            self.config['environment']['noise_level'] = noise_level
            for strategy in self.config['experiments']['coordination_strategies']:
                for run in range(self.config['experiments']['num_runs']):
                    result = self.run_experiment(strategy)
                    result['noise_level'] = noise_level
                    result['run_number'] = run
                    all_results.append(result)
        aggregated_results = self.metrics.aggregate_experiment_results(all_results)
        self._generate_visualizations(all_results, aggregated_results)
        self._save_results(all_results, aggregated_results)
    
    def _generate_visualizations(self, all_results: List[Dict[str, Any]],
                               aggregated_results: Dict[str, Any]) -> None:
        """
        generate and save experiment visualizations.
        """
        for agent_name, agent in self.agents.items():
            self.visualizer.plot_agent_metrics(
                agent.get_metrics(),
                f'{agent_name} performance metrics'
            )
            plt.savefig(os.path.join(self.results_dir, f'{agent_name}_metrics.png'))
            plt.close()
        coordination_results = []
        strategies = self.config['experiments']['coordination_strategies']
        for strategy in strategies:
            strategy_results = [r for r in all_results if r['coordination_strategy'] == strategy]
            coord_metrics = self.metrics.calculate_coordination_metrics(
                [r['agent_states'] for r in strategy_results]
            )
            coordination_results.append(coord_metrics)
        self.visualizer.plot_coordination_comparison(coordination_results, strategies)
        plt.savefig(os.path.join(self.results_dir, 'coordination_comparison.png'))
        plt.close()
        noise_results = []
        noise_levels = self.config['experiments']['noise_levels']
        for noise_level in noise_levels:
            noise_runs = [r for r in all_results if r['noise_level'] == noise_level]
            performance = self.metrics.calculate_classification_metrics(
                [True] * len(noise_runs),
                [r['results']['analyzer_results']['success_rate'] > 0.5 for r in noise_runs]
            )
            noise_results.append(performance)
        self.visualizer.plot_noise_impact(noise_levels, noise_results)
        plt.savefig(os.path.join(self.results_dir, 'noise_impact.png'))
        plt.close()
        self.visualizer.create_summary_report(
            aggregated_results,
            os.path.join(self.results_dir, 'experiment_summary.pdf')
        )
    
    def _save_results(self, all_results: List[Dict[str, Any]],
                     aggregated_results: Dict[str, Any]) -> None:
        """
        save experiment results to files.
        """
        with open(os.path.join(self.results_dir, 'all_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)
        with open(os.path.join(self.results_dir, 'aggregated_results.json'), 'w') as f:
            json.dump(aggregated_results, f, indent=2)
        with open(os.path.join(self.results_dir, 'config.json'), 'w') as f:
            json.dump(self.config, f, indent=2)

if __name__ == '__main__':
    runner = ExperimentRunner()
    runner.run_all_experiments() 