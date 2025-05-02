import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from main import run_pipeline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

strategies = ['sequential', 'parallel', 'hybrid']
noise_levels = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
results = []

for strategy in strategies:
    for noise in noise_levels:
        for run in range(3):  # Run each config 3 times for robustness
            res = run_pipeline(strategy=strategy, noise=noise)
            results.append(res)

df = pd.DataFrame(results)
df.to_csv('experiments/results/experiment_results.csv', index=False)
print(df.groupby(['strategy', 'noise'])[['precision', 'recall', 'f1', 'processing_time']].mean())

# Plot Precision/Recall/F1 by strategy and noise
for metric in ['precision', 'recall', 'f1']:
    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x='noise', y=metric, hue='strategy', marker='o')
    plt.title(f'{metric.capitalize()} by Noise Level and Strategy')
    plt.ylabel(metric.capitalize())
    plt.xlabel('Noise Level')
    plt.legend(title='Strategy')
    plt.tight_layout()
    plt.savefig(f'experiments/results/{metric}_by_noise_strategy.png')
    plt.close() 