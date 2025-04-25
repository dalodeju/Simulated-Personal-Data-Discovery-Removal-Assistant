"""
Experiment Coordinator for managing agent interactions and measuring performance.
Coordinates the multi-agent system and collects performance metrics.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime
import json
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..agents.web_scraper import WebScraperAgent, ScraperConfig
from ..agents.data_analyzer import DataAnalyzerAgent, AnalyzerConfig
from ..agents.risk_evaluator import RiskEvaluatorAgent, RiskEvaluatorConfig
from ..agents.recommender import RecommendationAgent, RecommenderConfig
from ..environment.ecosystem import DigitalEcosystem, EcosystemConfig

@dataclass
class ExperimentConfig:
    """Configuration for experiment coordination"""
    output_dir: str = 'experiments/results'
    num_iterations: int = 100
    coordination_strategies: List[str] = None
    noise_levels: List[float] = None
    
    def __post_init__(self):
        if self.coordination_strategies is None:
            self.coordination_strategies = [
                'sequential',
                'parallel',
                'hierarchical'
            ]
        if self.noise_levels is None:
            self.noise_levels = [0.0, 0.1, 0.2, 0.3]

class ExperimentCoordinator:
    """
    Coordinates the multi-agent system experiments and measures performance.
    Manages different coordination strategies and collects comprehensive metrics.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize environment and agents
        self.ecosystem = DigitalEcosystem(EcosystemConfig())
        self.scraper = WebScraperAgent(self.ecosystem, ScraperConfig())
        self.analyzer = DataAnalyzerAgent(AnalyzerConfig())
        self.evaluator = RiskEvaluatorAgent(RiskEvaluatorConfig())
        self.recommender = RecommendationAgent(RecommenderConfig())
        
        # Metrics storage
        self.metrics = {
            'discovery': [],
            'analysis': [],
            'risk': [],
            'recommendations': [],
            'overall': []
        }

    def _coordinate_sequential(self, 
                             initial_query: str) -> Dict[str, Any]:
        """Run agents in sequence"""
        try:
            # Web scraping phase
            scraping_start = datetime.now()
            discovered_data = self.scraper.deep_search(initial_query)
            scraping_time = (datetime.now() - scraping_start).total_seconds()
            
            # Data analysis phase
            analysis_start = datetime.now()
            analyzed_items = []
            for profile in discovered_data['profiles']:
                analysis = self.analyzer.analyze_content(profile)
                analyzed_items.append(analysis)
            analysis_time = (datetime.now() - analysis_start).total_seconds()
            
            # Risk evaluation phase
            evaluation_start = datetime.now()
            risk_assessments = []
            for item in analyzed_items:
                if 'error' not in item:
                    assessment = self.evaluator.evaluate_risk(item)
                    risk_assessments.append(assessment)
            evaluation_time = (datetime.now() - evaluation_start).total_seconds()
            
            # Recommendation phase
            recommendation_start = datetime.now()
            recommendations = []
            for assessment, analysis in zip(risk_assessments, analyzed_items):
                if 'error' not in assessment:
                    recommendation = self.recommender.generate_recommendations(
                        assessment, analysis
                    )
                    recommendations.append(recommendation)
            recommendation_time = (datetime.now() - recommendation_start).total_seconds()
            
            return {
                'discovered_data': discovered_data,
                'analyzed_items': analyzed_items,
                'risk_assessments': risk_assessments,
                'recommendations': recommendations,
                'timing': {
                    'scraping': scraping_time,
                    'analysis': analysis_time,
                    'evaluation': evaluation_time,
                    'recommendation': recommendation_time,
                    'total': scraping_time + analysis_time + 
                            evaluation_time + recommendation_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in sequential coordination: {str(e)}")
            return {'error': str(e)}

    def _coordinate_parallel(self, 
                           initial_query: str) -> Dict[str, Any]:
        """Simulate parallel agent execution"""
        try:
            # In a real parallel implementation, these would run concurrently
            # Here we simulate parallel execution by running independently
            
            # Web scraping phase (always runs first)
            scraping_start = datetime.now()
            discovered_data = self.scraper.deep_search(initial_query)
            scraping_time = (datetime.now() - scraping_start).total_seconds()
            
            # Parallel phases
            analysis_results = []
            risk_results = []
            recommendation_results = []
            
            max_parallel_time = 0
            
            # Process each profile independently
            for profile in discovered_data['profiles']:
                # Analysis branch
                analysis_start = datetime.now()
                analysis = self.analyzer.analyze_content(profile)
                analysis_time = (datetime.now() - analysis_start).total_seconds()
                analysis_results.append({
                    'result': analysis,
                    'time': analysis_time
                })
                
                # Only proceed with valid analysis results
                if 'error' not in analysis:
                    # Risk evaluation branch
                    evaluation_start = datetime.now()
                    assessment = self.evaluator.evaluate_risk(analysis)
                    evaluation_time = (datetime.now() - evaluation_start).total_seconds()
                    risk_results.append({
                        'result': assessment,
                        'time': evaluation_time
                    })
                    
                    # Recommendation branch
                    if 'error' not in assessment:
                        recommendation_start = datetime.now()
                        recommendation = self.recommender.generate_recommendations(
                            assessment, analysis
                        )
                        recommendation_time = (datetime.now() - recommendation_start).total_seconds()
                        recommendation_results.append({
                            'result': recommendation,
                            'time': recommendation_time
                        })
                
                # Track maximum parallel execution time
                branch_time = max(
                    analysis_time,
                    evaluation_time if risk_results else 0,
                    recommendation_time if recommendation_results else 0
                )
                max_parallel_time = max(max_parallel_time, branch_time)
            
            return {
                'discovered_data': discovered_data,
                'analyzed_items': [r['result'] for r in analysis_results],
                'risk_assessments': [r['result'] for r in risk_results],
                'recommendations': [r['result'] for r in recommendation_results],
                'timing': {
                    'scraping': scraping_time,
                    'parallel_processing': max_parallel_time,
                    'total': scraping_time + max_parallel_time
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in parallel coordination: {str(e)}")
            return {'error': str(e)}

    def _coordinate_hierarchical(self, 
                               initial_query: str) -> Dict[str, Any]:
        """Run agents in a hierarchical structure"""
        try:
            results = {
                'levels': [],
                'timing': {}
            }
            
            # Level 1: Web Scraping (Root)
            level1_start = datetime.now()
            discovered_data = self.scraper.deep_search(initial_query)
            results['levels'].append({
                'name': 'discovery',
                'data': discovered_data
            })
            results['timing']['level1'] = (datetime.now() - level1_start).total_seconds()
            
            # Level 2: Data Analysis
            level2_start = datetime.now()
            analyzed_items = []
            for profile in discovered_data['profiles']:
                analysis = self.analyzer.analyze_content(profile)
                if 'error' not in analysis:
                    analyzed_items.append(analysis)
            results['levels'].append({
                'name': 'analysis',
                'data': analyzed_items
            })
            results['timing']['level2'] = (datetime.now() - level2_start).total_seconds()
            
            # Level 3: Risk Evaluation and Recommendations (Parallel Branches)
            level3_start = datetime.now()
            risk_branch = []
            recommendation_branch = []
            
            for analysis in analyzed_items:
                # Risk Evaluation Branch
                assessment = self.evaluator.evaluate_risk(analysis)
                if 'error' not in assessment:
                    risk_branch.append(assessment)
                    
                    # Recommendations Sub-branch
                    recommendation = self.recommender.generate_recommendations(
                        assessment, analysis
                    )
                    if 'error' not in recommendation:
                        recommendation_branch.append(recommendation)
            
            results['levels'].append({
                'name': 'risk_evaluation',
                'data': risk_branch
            })
            results['levels'].append({
                'name': 'recommendations',
                'data': recommendation_branch
            })
            results['timing']['level3'] = (datetime.now() - level3_start).total_seconds()
            
            # Calculate total time
            results['timing']['total'] = sum(results['timing'].values())
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical coordination: {str(e)}")
            return {'error': str(e)}

    def _calculate_metrics(self, 
                          results: Dict[str, Any],
                          ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics for the experiment"""
        metrics = {}
        
        # Discovery metrics
        if 'discovered_data' in results:
            discovered = results['discovered_data']
            metrics['discovery_recall'] = len(discovered['profiles']) / len(ground_truth['profiles'])
            metrics['discovery_precision'] = sum(
                1 for p in discovered['profiles'] 
                if p['id'] in [gt['id'] for gt in ground_truth['profiles']]
            ) / len(discovered['profiles'])
        
        # Analysis metrics
        if 'analyzed_items' in results:
            analyzed = results['analyzed_items']
            valid_analyses = [a for a in analyzed if 'error' not in a]
            metrics['analysis_success_rate'] = len(valid_analyses) / len(analyzed)
            
            # Compare with ground truth categories
            predicted = [a['categories'].keys() for a in valid_analyses]
            actual = [gt['categories'] for gt in ground_truth['analyses']]
            precision, recall, f1, _ = precision_recall_fscore_support(
                actual, predicted, average='weighted'
            )
            metrics['analysis_precision'] = precision
            metrics['analysis_recall'] = recall
            metrics['analysis_f1'] = f1
        
        # Risk evaluation metrics
        if 'risk_assessments' in results:
            assessments = results['risk_assessments']
            valid_assessments = [a for a in assessments if 'error' not in a]
            metrics['risk_success_rate'] = len(valid_assessments) / len(assessments)
            
            # Compare risk levels with ground truth
            predicted_risks = [a['risk_level'] for a in valid_assessments]
            actual_risks = [gt['risk_level'] for gt in ground_truth['risks']]
            risk_precision, risk_recall, risk_f1, _ = precision_recall_fscore_support(
                actual_risks, predicted_risks, average='weighted'
            )
            metrics['risk_precision'] = risk_precision
            metrics['risk_recall'] = risk_recall
            metrics['risk_f1'] = risk_f1
        
        # Timing metrics
        if 'timing' in results:
            for timing_key, timing_value in results['timing'].items():
                metrics[f'time_{timing_key}'] = timing_value
        
        return metrics

    def run_experiment(self, 
                      coordination_strategy: str,
                      noise_level: float) -> Dict[str, Any]:
        """
        Run a single experiment with specified coordination strategy and noise level
        """
        try:
            # Configure environment for this experiment
            self.ecosystem.config.error_rate = noise_level
            
            # Initialize results storage
            results = {
                'config': {
                    'coordination_strategy': coordination_strategy,
                    'noise_level': noise_level
                },
                'iterations': []
            }
            
            # Run multiple iterations
            for i in range(self.config.num_iterations):
                self.logger.info(f"Running iteration {i+1}/{self.config.num_iterations}")
                
                # Select coordination strategy
                if coordination_strategy == 'sequential':
                    iteration_results = self._coordinate_sequential("john")
                elif coordination_strategy == 'parallel':
                    iteration_results = self._coordinate_parallel("john")
                elif coordination_strategy == 'hierarchical':
                    iteration_results = self._coordinate_hierarchical("john")
                else:
                    raise ValueError(f"Unknown coordination strategy: {coordination_strategy}")
                
                # Calculate metrics for this iteration
                metrics = self._calculate_metrics(
                    iteration_results,
                    self._get_ground_truth()
                )
                
                results['iterations'].append({
                    'iteration': i,
                    'results': iteration_results,
                    'metrics': metrics
                })
            
            # Calculate aggregate metrics
            results['aggregate_metrics'] = self._calculate_aggregate_metrics(results['iterations'])
            
            # Save results
            self._save_results(results, coordination_strategy, noise_level)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error running experiment: {str(e)}")
            return {'error': str(e)}

    def _calculate_aggregate_metrics(self, 
                                  iterations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all iterations"""
        metrics_df = pd.DataFrame([it['metrics'] for it in iterations])
        
        return {
            'mean': metrics_df.mean().to_dict(),
            'std': metrics_df.std().to_dict(),
            'min': metrics_df.min().to_dict(),
            'max': metrics_df.max().to_dict()
        }

    def _save_results(self,
                     results: Dict[str, Any],
                     strategy: str,
                     noise_level: float):
        """Save experiment results to file"""
        filename = f"results_{strategy}_{noise_level:.1f}.json"
        filepath = os.path.join(self.config.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

    def _get_ground_truth(self) -> Dict[str, Any]:
        """Load or generate ground truth data for evaluation"""
        # In a real implementation, this would load actual ground truth data
        # Here we're using a simplified version for demonstration
        return {
            'profiles': [
                {'id': f"profile_{i}", 'categories': ['financial', 'personal_id']}
                for i in range(10)
            ],
            'analyses': [
                {'categories': ['financial', 'personal_id']}
                for _ in range(10)
            ],
            'risks': [
                {'risk_level': 'high'}
                for _ in range(10)
            ]
        }

    def visualize_results(self):
        """Generate visualizations of experiment results"""
        # Create results directory if it doesn't exist
        vis_dir = os.path.join(self.config.output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # Collect all results
        all_results = []
        for strategy in self.config.coordination_strategies:
            for noise_level in self.config.noise_levels:
                filename = f"results_{strategy}_{noise_level:.1f}.json"
                filepath = os.path.join(self.config.output_dir, filename)
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        results = json.load(f)
                        all_results.append({
                            'strategy': strategy,
                            'noise_level': noise_level,
                            'metrics': results['aggregate_metrics']['mean']
                        })
        
        if not all_results:
            self.logger.warning("No results found to visualize")
            return
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Plot performance metrics
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=results_df.melt(
                id_vars=['strategy', 'noise_level'],
                value_vars=['discovery_precision', 'analysis_f1', 'risk_f1']
            ),
            x='strategy',
            y='value',
            hue='variable'
        )
        plt.title('Performance Metrics by Coordination Strategy')
        plt.savefig(os.path.join(vis_dir, 'performance_metrics.png'))
        plt.close()
        
        # Plot timing metrics
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=results_df,
            x='noise_level',
            y='time_total',
            hue='strategy'
        )
        plt.title('Total Processing Time vs. Noise Level')
        plt.savefig(os.path.join(vis_dir, 'timing_metrics.png'))
        plt.close()
        
        # Plot noise impact
        plt.figure(figsize=(12, 6))
        for metric in ['discovery_precision', 'analysis_f1', 'risk_f1']:
            sns.lineplot(
                data=results_df,
                x='noise_level',
                y=metric,
                label=metric
            )
        plt.title('Impact of Noise on Different Metrics')
        plt.savefig(os.path.join(vis_dir, 'noise_impact.png'))
        plt.close()

def main():
    """Main function to run experiments"""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize coordinator
    config = ExperimentConfig()
    coordinator = ExperimentCoordinator(config)
    
    # Run experiments for each combination
    for strategy in config.coordination_strategies:
        for noise_level in config.noise_levels:
            print(f"\nRunning experiment with strategy={strategy}, noise={noise_level}")
            results = coordinator.run_experiment(strategy, noise_level)
            
            if 'error' not in results:
                print("Experiment completed successfully")
                print("Aggregate Metrics:")
                for metric, value in results['aggregate_metrics']['mean'].items():
                    print(f"- {metric}: {value:.3f}")
            else:
                print(f"Experiment failed: {results['error']}")
    
    # Generate visualizations
    coordinator.visualize_results()

if __name__ == '__main__':
    main() 