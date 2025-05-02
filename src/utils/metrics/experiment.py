"""
Metrics module for measuring and analyzing agent performance in experiments.
Provides tools for calculating, aggregating, and visualizing performance metrics.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import os

class PerformanceMetrics:
    """
    Handles calculation and analysis of performance metrics for the agent system.
    Includes accuracy, timing, and efficiency metrics.
    """
    
    def __init__(self):
        self.metrics_history = {
            'discovery': [],
            'analysis': [],
            'risk': [],
            'recommendations': [],
            'timing': [],
            'system': []
        }

    def calculate_discovery_metrics(self, 
                                  discovered: List[Dict[str, Any]],
                                  ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for the discovery phase"""
        if not discovered or not ground_truth:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }
            
        # Calculate precision
        true_positives = sum(
            1 for item in discovered 
            if any(gt['id'] == item['id'] for gt in ground_truth)
        )
        precision = true_positives / len(discovered) if discovered else 0.0
        
        # Calculate recall
        recall = true_positives / len(ground_truth) if ground_truth else 0.0
        
        # Calculate F1 score
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if precision + recall > 0 else 0.0
        )
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_discovered': len(discovered),
            'total_ground_truth': len(ground_truth)
        }

    def calculate_analysis_metrics(self,
                                 predictions: List[Dict[str, Any]],
                                 ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for the analysis phase"""
        if not predictions or not ground_truth:
            return {
                'accuracy': 0.0,
                'error_rate': 0.0
            }
            
        # Prepare data for sklearn metrics
        y_true = [gt['categories'] for gt in ground_truth]
        y_pred = [p['categories'] for p in predictions]
        
        # Calculate precision, recall, and F1 for each category
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Calculate error rate
        error_count = sum(
            1 for p in predictions if 'error' in p
        )
        error_rate = error_count / len(predictions)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'error_rate': error_rate,
            'processed_items': len(predictions)
        }

    def calculate_risk_metrics(self,
                             assessments: List[Dict[str, Any]],
                             ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for the risk evaluation phase"""
        if not assessments or not ground_truth:
            return {
                'accuracy': 0.0,
                'risk_correlation': 0.0
            }
            
        # Extract risk levels and scores
        true_levels = [gt['risk_level'] for gt in ground_truth]
        pred_levels = [a['risk_level'] for a in assessments]
        
        # Calculate classification metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_levels, pred_levels, average='weighted'
        )
        
        # Calculate risk score correlation
        true_scores = [float(gt.get('risk_score', 0.0)) for gt in ground_truth]
        pred_scores = [float(a.get('risk_score', 0.0)) for a in assessments]
        correlation = np.corrcoef(true_scores, pred_scores)[0, 1]
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'risk_correlation': correlation,
            'assessed_items': len(assessments)
        }

    def calculate_recommendation_metrics(self,
                                      recommendations: List[Dict[str, Any]],
                                      ground_truth: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate metrics for the recommendation phase"""
        if not recommendations or not ground_truth:
            return {
                'relevance_score': 0.0,
                'coverage': 0.0
            }
            
        # Calculate recommendation relevance
        relevance_scores = []
        for rec, gt in zip(recommendations, ground_truth):
            if 'recommendations' in rec and 'expected_actions' in gt:
                matched_actions = set(rec['recommendations']) & set(gt['expected_actions'])
                relevance = len(matched_actions) / len(gt['expected_actions'])
                relevance_scores.append(relevance)
        
        avg_relevance = np.mean(relevance_scores) if relevance_scores else 0.0
        
        # Calculate coverage
        total_issues = sum(len(gt['expected_actions']) for gt in ground_truth)
        total_recommendations = sum(
            len(rec['recommendations']) for rec in recommendations
        )
        coverage = total_recommendations / total_issues if total_issues > 0 else 0.0
        
        return {
            'relevance_score': avg_relevance,
            'coverage': coverage,
            'total_recommendations': total_recommendations,
            'unique_recommendations': len(set(
                r for rec in recommendations 
                for r in rec.get('recommendations', [])
            ))
        }

    def calculate_timing_metrics(self, 
                               timing_data: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate timing and efficiency metrics"""
        if not timing_data:
            return {
                'avg_processing_time': 0.0,
                'throughput': 0.0
            }
            
        # Calculate average processing times
        processing_times = [t['total'] for t in timing_data]
        avg_time = np.mean(processing_times)
        std_time = np.std(processing_times)
        
        # Calculate throughput (items per second)
        total_items = sum(t.get('items_processed', 1) for t in timing_data)
        total_time = sum(processing_times)
        throughput = total_items / total_time if total_time > 0 else 0.0
        
        return {
            'avg_processing_time': avg_time,
            'std_processing_time': std_time,
            'throughput': throughput,
            'total_time': total_time,
            'total_items': total_items
        }

    def calculate_system_metrics(self, 
                               system_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall system performance metrics"""
        metrics = {}
        
        # Calculate success rate
        total_operations = system_data.get('total_operations', 0)
        failed_operations = system_data.get('failed_operations', 0)
        if total_operations > 0:
            metrics['success_rate'] = (
                total_operations - failed_operations
            ) / total_operations
        else:
            metrics['success_rate'] = 0.0
            
        # Calculate resource utilization
        metrics['cpu_utilization'] = system_data.get('cpu_usage', 0.0)
        metrics['memory_utilization'] = system_data.get('memory_usage', 0.0)
        
        # Calculate coordination efficiency
        coordination_delays = system_data.get('coordination_delays', [])
        if coordination_delays:
            metrics['avg_coordination_delay'] = np.mean(coordination_delays)
            metrics['max_coordination_delay'] = np.max(coordination_delays)
        else:
            metrics['avg_coordination_delay'] = 0.0
            metrics['max_coordination_delay'] = 0.0
        
        return metrics

    def update_metrics(self, 
                      phase: str,
                      metrics: Dict[str, float]) -> None:
        """Update metrics history with new measurements"""
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history[phase].append(metrics)

    def get_aggregate_metrics(self, 
                            phase: str,
                            window: Optional[int] = None) -> Dict[str, float]:
        """Calculate aggregate metrics over a time window"""
        if phase not in self.metrics_history:
            return {}
            
        # Get metrics for the specified window
        metrics_list = self.metrics_history[phase]
        if window is not None:
            metrics_list = metrics_list[-window:]
            
        if not metrics_list:
            return {}
            
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(metrics_list)
        
        # Calculate aggregates for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        aggregates = {
            col: {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
            for col in numeric_cols
        }
        
        return aggregates

    def plot_metrics_over_time(self, 
                             phase: str,
                             metric_names: List[str],
                             output_dir: str) -> None:
        """Plot metrics trends over time"""
        if phase not in self.metrics_history:
            return
            
        metrics_list = self.metrics_history[phase]
        if not metrics_list:
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_list)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create plot
        plt.figure(figsize=(12, 6))
        for metric in metric_names:
            if metric in df.columns:
                plt.plot(df['timestamp'], df[metric], label=metric)
                
        plt.title(f'{phase.capitalize()} Metrics Over Time')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'{phase}_metrics.png'))
        plt.close()

    def plot_correlation_matrix(self, 
                              phase: str,
                              output_dir: str) -> None:
        """Plot correlation matrix of metrics"""
        if phase not in self.metrics_history:
            return
            
        metrics_list = self.metrics_history[phase]
        if not metrics_list:
            return
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_list)
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return
            
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, 
                   annot=True,
                   cmap='coolwarm',
                   center=0,
                   fmt='.2f')
        
        plt.title(f'{phase.capitalize()} Metrics Correlation Matrix')
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f'{phase}_correlation.png'))
        plt.close()

    def save_metrics(self, 
                    output_file: str) -> None:
        """Save metrics history to file"""
        with open(output_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)

    def load_metrics(self, 
                    input_file: str) -> None:
        """Load metrics history from file"""
        with open(input_file, 'r') as f:
            self.metrics_history = json.load(f) 