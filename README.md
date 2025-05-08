# Simulated Personal Data Discovery & Removal Assistant

This project builds and evaluates a multi‑agent system for finding, analyzing, and recommending actions on personal data in a controlled, simulated digital environment. Data privacy is a growing concern. Companies, regulators, and individuals need tools to locate exposed personal information, assess its risk, and decide how to deal with that exposure respectively. A multi‑agent approach allows specialized components to focus on discovery, classification and risk scoring to generate recommendations for the next course of action.

## Project Overview

This project simulates a digital environment with synthetic user profiles, posts, and articles. 

- **Web Scraper Agent**: Locates digital traces in the ecosystem
- **Data Analyzer Agent**: Uses NLP to classify sensitive data
- **Risk Evaluation Agent**: Assigns risk scores based on content features
- **Recommendation Agent**: Recommends steps for data removal or protection

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

## Running the Main Pipeline
To run the full agent pipeline on a sample input and see how all agents work together, use:
```bash
python src/main.py
```
This will:
- Run the pipeline end-to-end (discovery, analysis, risk evaluation, recommendations)
- Use a sample text containing personal data
- Print a summary of results, risk levels, recommendations, and performance metrics to the terminal
- Save a detailed JSON output to `experiments/results/` (e.g., `scan_results_YYYYMMDD_HHMMSS.json`)

### Outputs

- **Terminal**: Summary of discovered data, risk levels, recommendations is printed
- **JSON**: `scan_results_YYYYMMDD_HHMMSS.json` is saved in `experiments/results/`

## Running Experiments

To run the main experiment pipeline and generate results:
```bash
python experiments/run_experiments.py
```
This will:
- Run experiments across multiple agent strategies, risk methods, and noise levels
- Save results and plots to `experiments/results/`
- Print summary tables for precision, recall, F1, and total processing time

### Outputs

- **Terminal**: Resulting tables for each metric is printed
- **CSV**: All raw results in `experiment_results.csv`
- **Figures**: Saved in `experiments/results/`: 
  - Precision/Recall/F1 vs. Coordination Strategy
  - Precision/Recall/F1 vs. Risk Evaluation Method
  - F1 vs. Noise Level
  - Processing Time vs. Risk Method

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

## Future Work

- Real-time or online agent coordination
- More advanced NLP and risk models
- Web-based results dashboard
- Support for additional data types and languages
