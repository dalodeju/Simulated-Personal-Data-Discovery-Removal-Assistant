# simulated personal data discovery & removal assistant

this project is a master's-level simulation of a digital ecosystem, designed to explore how intelligent agents can discover, analyze, and recommend actions for personal data found in synthetic user profiles, posts, and articles. the system is modular, extensible, and built for robust experimentation and evaluation.

---

## overview

the system simulates a digital world with synthetic user data and four main agents:
- web scraper agent: finds personal data in text using pattern matching.
- data analyzer agent: classifies and analyzes the sensitivity and context of discovered data.
- risk evaluator agent: assesses the risk associated with each data item using configurable criteria.
- recommendation agent: suggests actions for data protection and removal.

the project supports automated experiments, multiple agent coordination strategies, noise injection, and detailed performance metrics (precision, recall, f1, processing time, and more).

---

## key features

- synthetic data generation: realistic, diverse, and noisy user profiles and posts.
- agent coordination: supports sequential, parallel, and hybrid agent strategies.
- risk evaluation: flexible, multi-factor risk scoring.
- metrics & visualization: automated experiment runs, csv export, and plotting.
- extensible design: easy to add new agents, strategies, or data types.

---

## project structure

main directories and files:
- `src/agents/` — agent implementations (web_scraper.py, data_analyzer.py, risk_evaluator.py, recommender.py)
- `src/environment/` — digital ecosystem simulation and data generation
- `src/experiments/` — experiment runner and coordinator
- `src/utils/` — utility functions and metrics modules
- `models/` — trained models (e.g., analyzer models)
- `experiments/data/` — synthetic datasets
- `experiments/results/` — experiment outputs and plots
- `tests/` — test suite for agents and integration
- `config.yaml` — main configuration file
- `requirements.txt` — python dependencies
- `README.md` — project documentation

---

## installation

1. clone the repository:
   ```bash
   git clone <repository-url>
   cd <project-directory>
   ```

2. set up a virtual environment:
   ```bash
   python -m venv .venv
   # on windows:
   .venv\Scripts\activate
   # on unix/mac:
   source .venv/bin/activate
   ```

3. install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. download required nlp models:
   ```bash
   python -m spacy download en_core_web_sm
   ```

---

## running experiments

to run the full experiment pipeline with all strategies and noise levels:
```bash
python -m src.experiments.run_experiments
```
results, metrics, and plots will be saved in the `experiments/results/` directory.

---

## configuration

all system and agent settings are managed in `config.yaml`. you can adjust:
- number of profiles, noise levels, and sensitive data ratio
- agent-specific thresholds and model paths
- experiment strategies and number of runs

---

## testing

to run all tests:
```bash
python -m pytest tests/
```
you can also run specific test files, for example:
```bash
python -m pytest tests/test_agents/test_data_analyzer.py
```

---

## metrics & evaluation

the system automatically calculates and saves:
- precision, recall, f1 for discovery, analysis, and risk evaluation
- processing time and throughput
- recommendation relevance and coverage
- plots for performance trends and noise impact

all results are saved as csv and png files for easy analysis.

---

## extending the project

- add new agent strategies: implement a new method in the experiment runner and add it to the config.
- change data generation: edit `src/environment/data_generator.py` for new data types or noise models.
- customize risk evaluation: adjust weights and thresholds in `src/agents/risk_evaluator.py` or the config.

---

## future work

- real-time or streaming data simulation
- web-based dashboard for results
- more advanced nlp and ml models
- support for additional languages and data types

---

## contact

for questions or collaboration, please contact the project maintainer.