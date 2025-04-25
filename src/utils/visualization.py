from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class ExperimentVisualizer:
    """Utility class for visualizing experiment results."""
    
    @staticmethod
    def plot_agent_metrics(agent_metrics: Dict[str, List[float]], title: str = None) -> None:
        """Plot agent performance metrics over time.
        
        Args:
            agent_metrics: Dictionary of metric lists from an agent
            title: Optional title for the plot
        """
        plt.figure(figsize=(12, 6))
        
        for metric_name, values in agent_metrics.items():
            plt.plot(values, label=metric_name)
        
        plt.xlabel('Time Step')
        plt.ylabel('Metric Value')
        plt.title(title or 'Agent Performance Metrics')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    
    @staticmethod
    def plot_coordination_comparison(coordination_results: List[Dict[str, float]],
                                  strategies: List[str]) -> None:
        """Plot comparison of different coordination strategies.
        
        Args:
            coordination_results: List of coordination metric dictionaries
            strategies: List of strategy names
        """
        metrics = ['total_processing_time', 'avg_success_rate', 'avg_error_rate']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        fig.suptitle('Coordination Strategy Comparison')
        
        for i, metric in enumerate(metrics):
            values = [result[metric] for result in coordination_results]
            axes[i].bar(strategies, values)
            axes[i].set_title(metric.replace('_', ' ').title())
            axes[i].set_xticklabels(strategies, rotation=45)
        
        plt.tight_layout()
    
    @staticmethod
    def plot_noise_impact(noise_levels: List[float],
                         performance_metrics: List[Dict[str, float]]) -> None:
        """Plot the impact of noise on system performance.
        
        Args:
            noise_levels: List of noise levels tested
            performance_metrics: List of performance metric dictionaries
        """
        metrics = ['precision', 'recall', 'f1']
        
        plt.figure(figsize=(10, 6))
        
        for metric in metrics:
            values = [metrics[metric] for metrics in performance_metrics]
            plt.plot(noise_levels, values, marker='o', label=metric)
        
        plt.xlabel('Noise Level')
        plt.ylabel('Metric Value')
        plt.title('Impact of Noise on Performance')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    
    @staticmethod
    def plot_efficiency_heatmap(processing_times: List[float],
                              success_rates: List[float],
                              conditions: List[str]) -> None:
        """Create a heatmap of efficiency metrics across different conditions.
        
        Args:
            processing_times: List of processing times
            success_rates: List of success rates
            conditions: List of condition labels
        """
        # Create efficiency matrix
        efficiency_scores = []
        for time, rate in zip(processing_times, success_rates):
            time_norm = 1 - (time / max(processing_times))
            efficiency = time_norm * 0.4 + rate * 0.6
            efficiency_scores.append(efficiency)
        
        # Reshape data for heatmap
        matrix_size = int(np.sqrt(len(conditions)))
        efficiency_matrix = np.array(efficiency_scores).reshape(matrix_size, matrix_size)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(efficiency_matrix,
                   annot=True,
                   fmt='.2f',
                   cmap='YlOrRd',
                   xticklabels=conditions[:matrix_size],
                   yticklabels=conditions[:matrix_size])
        
        plt.title('Efficiency Heatmap')
        plt.tight_layout()
    
    @staticmethod
    def plot_experiment_summary(experiment_results: Dict[str, Any]) -> None:
        """Create a summary visualization of experiment results.
        
        Args:
            experiment_results: Dictionary of aggregated experiment results
        """
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(2, 2)
        
        # Plot 1: Overall metrics
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = experiment_results['metrics']
        metric_names = [name for name in metrics.keys() if name.endswith('_mean')]
        metric_values = [metrics[name] for name in metric_names]
        ax1.bar(range(len(metric_names)), metric_values)
        ax1.set_xticks(range(len(metric_names)))
        ax1.set_xticklabels([name.replace('_mean', '') for name in metric_names], rotation=45)
        ax1.set_title('Average Performance Metrics')
        
        # Plot 2: Coordination metrics
        ax2 = fig.add_subplot(gs[0, 1])
        coord_metrics = experiment_results['coordination']
        ax2.bar(coord_metrics.keys(), coord_metrics.values())
        ax2.set_xticklabels(coord_metrics.keys(), rotation=45)
        ax2.set_title('Coordination Metrics')
        
        # Plot 3: Efficiency metrics
        ax3 = fig.add_subplot(gs[1, :])
        eff_metrics = experiment_results['efficiency']
        ax3.bar(eff_metrics.keys(), eff_metrics.values())
        ax3.set_title('Efficiency Metrics')
        
        plt.tight_layout()
    
    @staticmethod
    def plot_risk_distribution(risk_scores: List[float], thresholds: List[float]) -> None:
        """Plot the distribution of risk scores with threshold lines.
        
        Args:
            risk_scores: List of risk scores
            thresholds: List of threshold values to mark
        """
        plt.figure(figsize=(10, 6))
        
        # Plot risk score distribution
        sns.histplot(risk_scores, bins=30, kde=True)
        
        # Add threshold lines
        for threshold in thresholds:
            plt.axvline(x=threshold, color='r', linestyle='--',
                       label=f'Threshold: {threshold:.2f}')
        
        plt.xlabel('Risk Score')
        plt.ylabel('Count')
        plt.title('Distribution of Risk Scores')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    
    @staticmethod
    def create_summary_report(experiment_results: Dict[str, Any],
                            output_file: str = 'experiment_summary.pdf') -> None:
        """Create a comprehensive summary report of experiment results.
        
        Args:
            experiment_results: Dictionary of aggregated experiment results
            output_file: Path to save the PDF report
        """
        # Create a multi-page figure
        with PdfPages(output_file) as pdf:
            # Page 1: Overall metrics
            ExperimentVisualizer.plot_experiment_summary(experiment_results)
            pdf.savefig()
            plt.close()
            
            # Page 2: Detailed metrics
            if 'detailed_metrics' in experiment_results:
                fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                fig.suptitle('Detailed Performance Analysis')
                
                metrics = experiment_results['detailed_metrics']
                for i, (metric, values) in enumerate(metrics.items()):
                    row = i // 2
                    col = i % 2
                    axes[row, col].plot(values)
                    axes[row, col].set_title(metric)
                    axes[row, col].grid(True)
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
            
            # Page 3: Statistical analysis
            if 'statistical_tests' in experiment_results:
                fig, ax = plt.subplots(figsize=(10, 6))
                stats = experiment_results['statistical_tests']
                
                ax.table(cellText=[[k, f"{v:.4f}"] for k, v in stats.items()],
                        colLabels=['Metric', 'P-Value'],
                        loc='center')
                ax.axis('off')
                
                plt.title('Statistical Analysis Results')
                plt.tight_layout()
                pdf.savefig()
                plt.close() 