# Simulated Personal Data Discovery & Removal Assistant

A robust multi-agent system for discovering, analyzing, and recommending actions on personal data in a simulated digital ecosystem. Developed as a Master's project for Designing Intelligent Agents.

## Project Overview

This project simulates a digital environment with synthetic user profiles, posts, and articles. Four specialized agents work together to:
- Discover digital traces of personal data
- Analyze and classify sensitive information using NLP
- Evaluate risk based on content features
- Recommend actions for data removal or protection

The system supports experiments to analyze:
- The effect of agent coordination on discovery accuracy
- The impact of risk evaluation methods on speed and accuracy
- The robustness of agents under varying data noise

## Features

- Modular, extensible agent architecture
- Synthetic, parameterizable digital ecosystem
- Multiple agent coordination strategies
- Configurable risk evaluation methods
- Precision, recall, F1, and timing metrics
- Automated experiment pipeline and result visualization

## Installation

1. Clone the repository:
   ```bash
   git clone <dalodeju/Simulated-Personal-Data-Discovery-Removal-Assistant>
   cd <dalodeju/Simulated-Personal-Data-Discovery-Removal-Assistant>
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate 
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Download required models:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Configuration

- All main settings are in `config.yaml` (agent parameters, risk weights, etc.)
- Data files are in `experiments/data/`
- Trained models are in `models/analyzer/`

## Project Structure

- `src/`
  - `main.py` — Main pipeline and experiment runner
  - `agents/`
    - `base_agent.py` — Abstract agent class
    - `web_scraper.py` — Web Scraper Agent
    - `data_analyzer.py` — Data Analyzer Agent (NLP)
    - `risk_evaluator.py` — Risk Evaluation Agent
    - `recommender.py` — Recommendation Agent
  - `environment/`
    - `ecosystem.py` — Digital ecosystem and profile generator
    - `data_generator.py` — Synthetic data generation
  - `utils/`
    - `common.py` — Shared utilities
    - `metrics/`
      - `base.py` — Basic metrics
      - `experiment.py` — Advanced experiment metrics
- `experiments/`
  - `run_experiments.py` — Main experiment script
  - `data/` — Synthetic profiles and posts
  - `results/` — Experiment outputs (plots, CSVs)
- `models/`
  - `analyzer/` — Trained model files (`vectorizer.joblib`, `classifier.joblib`)
- `requirements.txt` — Python dependencies
- `config.yaml` — Main configuration file
- `README.md` — Project documentation
- `.gitignore` — Git ignore rules

## Running Experiments

To run the main experiment pipeline and generate results:
```bash
python experiments/run_experiments.py
```
This will:
- Run experiments across multiple agent strategies, risk methods, and noise levels
- Save results and plots to `experiments/results/`
- Print summary tables for precision, recall, F1, and timing

## Interpreting Results

- **Summary tables**: Printed in the terminal for each research question
- **Plots**: Saved in `experiments/results/` (e.g., F1 vs. noise, strategy comparisons)
- **CSV**: All raw results in `experiment_results.csv`

Key plots include:
- Precision/Recall/F1 by coordination strategy
- F1 vs. noise level
- F1 by risk evaluation method
- Processing time by risk method

## Agents

- **Web Scraper Agent**: Locates digital traces in the ecosystem
- **Data Analyzer Agent**: Uses NLP to classify sensitive data
- **Risk Evaluation Agent**: Assigns risk scores based on content features
- **Recommendation Agent**: Recommends steps for data removal or protection

## Future Work

- Real-time or online agent coordination
- More advanced NLP and risk models
- Web-based results dashboard
- Support for additional data types and languages
