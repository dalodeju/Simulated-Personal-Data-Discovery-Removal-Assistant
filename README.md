# Simulated Personal Data Discovery & Removal Assistant

This project builds and evaluates a multi‑agent system for finding, analyzing, and recommending actions on personal data in a controlled, simulated digital environment. Data privacy is a growing concern: companies, regulators, and individuals need tools to locate exposed personal information, assess its risk, and decide how to mitigate exposure. A multi‑agent approach lets specialized components focus on discovery, classification, risk scoring, and recommendations, then coordinate to produce a unified workflow.

## Project Overview

This project simulates a digital environment with synthetic user profiles, posts, and articles. 

- **Web Scraper Agent**: Locates digital traces in the ecosystem
- **Data Analyzer Agent**: Uses NLP to classify sensitive data
- **Risk Evaluation Agent**: Assigns risk scores based on content features
- **Recommendation Agent**: Recommends steps for data removal or protection

## Features

- Parameterizable digital ecosystem
- Multiple agent coordination strategies
- Configurable risk evaluation methods
- Precision, recall, F1, and timing metrics
- Automated experiment pipeline and result visualization

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dalodeju/simulated-personal-data-discovery-removal-assistant.git
   cd simulated-personal-data-discovery-removal-assistant
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
  - `main.py` - Main pipeline
  - `agents/`
    - `base_agent.py` - Base Agent Class
    - `web_scraper.py` - Web Scraper Agent
    - `data_analyzer.py` - Data Analyzer Agent (NLP)
    - `risk_evaluator.py` - Risk Evaluation Agent
    - `recommender.py` - Recommendation Agent
  - `environment/`
    - `ecosystem.py` - Digital ecosystem and profile generator
    - `data_generator.py` - Synthetic data generation
  - `utils/`
    - `common.py` - Shared utilities
    - `metrics/`
      - `base.py` - Basic metrics
      - `experiment.py` - Advanced experiment metrics
- `experiments/`
  - `run_experiments.py` - Main experiment script
  - `data/` - Generated profiles and posts
  - `results/` - Experiment outputs (figures, CSVs)
- `models/`
  - `analyzer/` - Trained model files (`vectorizer.joblib`, `classifier.joblib`)
- `requirements.txt` - Dependencies
- `config.yaml` - Main config file
- `README.md` - Project docs
- `.gitignore` - Git ignore

## Running Experiments

To run the main experiment pipeline and generate results:
```bash
python experiments/run_experiments.py
```
This will:
- Run experiments across multiple agent strategies, risk methods, and noise levels
- Save results and plots to `experiments/results/`
- Print summary tables for precision, recall, F1, and total processing time

## Interpreting Results

- **Summary tables**: Printed in the terminal for each research question
- **Figures**: Saved in `experiments/results/` (e.g., F1 vs. noise, strategy comparisons)
- **CSV**: All raw results in `experiment_results.csv`

Figures include:
- Precision/Recall/F1 by coordination strategy
- F1 vs. noise level
- F1 by risk evaluation method
- Processing time by risk method

## Future Work

- Real-time or online agent coordination
- More advanced NLP and risk models
- Web-based results dashboard
- Support for additional data types and languages
