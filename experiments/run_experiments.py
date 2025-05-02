import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from main import run_pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define experimental variables
strategies = ['sequential', 'parallel', 'hybrid']
noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
risk_methods = {
    'default': None,  # Use default config
    'exposure-heavy': {
        'weights': {
            'sensitivity': 0.15,
            'exposure': 0.55,
            'freshness': 0.10,
            'combination': 0.10,
            'context': 0.10
        }
    },
    'sensitivity-heavy': {
        'weights': {
            'sensitivity': 0.60,
            'exposure': 0.10,
            'freshness': 0.10,
            'combination': 0.10,
            'context': 0.10
        }
    }
}

results = []

# Run experiments
for strategy in strategies:
    for risk_method, risk_config in risk_methods.items():
        for noise in noise_levels:
            for run in range(3):  # Run each config 3 times for robustness
                res = run_pipeline(strategy=strategy, noise=noise, risk_config=risk_config)
                res['risk_method'] = risk_method
                results.append(res)

# Collect results
df = pd.DataFrame(results)
df.to_csv('experiments/results/experiment_results.csv', index=False)

# 1. Agent Coordination vs. Accuracy
print("\n=== Agent Coordination vs. Data Discovery Accuracy ===")
coord_summary = df.groupby(['strategy'])[['precision', 'recall', 'f1']].mean().reset_index()
print(coord_summary)
plt.figure(figsize=(8, 5))
sns.barplot(data=coord_summary.melt(id_vars='strategy'), x='strategy', y='value', hue='variable')
plt.title('Precision, Recall, F1 by Coordination Strategy')
plt.ylabel('Score')
plt.xlabel('Coordination Strategy')
plt.legend(title='Metric')
plt.tight_layout()
plt.savefig('experiments/results/coordination_vs_accuracy.png')
plt.close()

# 2. Risk Evaluation Method Impact
print("\n=== Risk Evaluation Method Impact ===")
risk_summary = df.groupby(['risk_method'])[['precision', 'recall', 'f1', 'processing_time']].mean().reset_index()
print(risk_summary)
plt.figure(figsize=(8, 5))
sns.barplot(data=risk_summary.melt(id_vars='risk_method', value_vars=['precision', 'recall', 'f1']), x='risk_method', y='value', hue='variable')
plt.title('Precision, Recall, F1 by Risk Evaluation Method')
plt.ylabel('Score')
plt.xlabel('Risk Evaluation Method')
plt.legend(title='Metric')
plt.tight_layout()
plt.savefig('experiments/results/risk_method_vs_accuracy.png')
plt.close()
plt.figure(figsize=(8, 5))
sns.barplot(data=risk_summary, x='risk_method', y='processing_time')
plt.title('Processing Time by Risk Evaluation Method')
plt.ylabel('Processing Time (s)')
plt.xlabel('Risk Evaluation Method')
plt.tight_layout()
plt.savefig('experiments/results/risk_method_vs_time.png')
plt.close()

# 3. Robustness to Noise
print("\n=== Agent Robustness to Data Noise ===")
noise_summary = df.groupby(['noise'])[['precision', 'recall', 'f1']].mean().reset_index()
print(noise_summary)
plt.figure(figsize=(8, 5))
sns.lineplot(data=noise_summary, x='noise', y='f1', marker='o')
plt.title('F1 Score vs. Noise Level')
plt.ylabel('F1 Score')
plt.xlabel('Noise Level')
plt.tight_layout()
plt.savefig('experiments/results/f1_vs_noise.png')
plt.close()

# Additional: F1 by Noise and Strategy
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='noise', y='f1', hue='strategy', marker='o')
plt.title('F1 Score by Noise Level and Strategy')
plt.ylabel('F1 Score')
plt.xlabel('Noise Level')
plt.legend(title='Strategy')
plt.tight_layout()
plt.savefig('experiments/results/f1_by_noise_strategy.png')
plt.close()

# Additional: F1 by Noise and Risk Method
plt.figure(figsize=(8, 5))
sns.lineplot(data=df, x='noise', y='f1', hue='risk_method', marker='o')
plt.title('F1 Score by Noise Level and Risk Method')
plt.ylabel('F1 Score')
plt.xlabel('Noise Level')
plt.legend(title='Risk Method')
plt.tight_layout()
plt.savefig('experiments/results/f1_by_noise_risk_method.png')
plt.close()

print("\nAll results and plots saved to experiments/results/") 