# Personal Data Discovery Agent

A focused intelligent agent system for discovering and analyzing personal data in digital environments, developed as part of a Master's project in Designing Intelligent Agents.

## Project Overview

This project implements an intelligent agent system that can:
- Discover personal data (emails, phone numbers, SSNs, credit cards) in text
- Analyze data sensitivity and context using NLP
- Evaluate risk levels using configurable criteria
- Generate detailed analysis reports and performance metrics
- Provide data protection recommendations

## Features

### Core Functionality
- Pattern-based personal data detection using regex
- NLP-based context analysis using spaCy and Transformers
- Machine learning-based sensitivity classification
- Configurable risk assessment with multiple factors
- Comprehensive performance metrics and visualization
- Detailed logging and reporting

### Technical Features
- Modular agent architecture with clear separation of concerns
- Advanced metrics system for performance tracking
- Comprehensive test suite with 42 test cases
- YAML-based configuration
- Efficient batch processing capabilities

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
source .venv/bin/activate  # On Unix/MacOS
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

The project uses a YAML configuration file (`config.yaml`) for all settings:

### Agent Configuration
- Web scraper settings (retries, timeouts, confidence thresholds)
- Data analyzer settings (model paths, batch size, GPU usage)
- Risk evaluator settings (risk thresholds, category weights)
- Recommender settings (priority levels, action types)

### System Configuration
- Logging configuration
- Performance metrics settings
- Output directories
- Model paths

## Project Structure

```
src/
├── agents/                # Agent implementations
│   ├── base_agent.py     # Abstract base agent class
│   ├── web_scraper.py    # Web scraping agent
│   ├── data_analyzer.py  # Data analysis agent
│   ├── risk_evaluator.py # Risk assessment agent
│   └── recommender.py    # Recommendation agent
├── environment/          # Test environment
├── experiments/         # Experiment configurations and results
│   ├── data/           # Test datasets
│   └── results/        # Experiment outputs
├── utils/              # Utility functions
│   ├── metrics/        # Performance metrics
│   │   ├── base.py    # Basic metrics
│   │   └── experiment.py # Advanced metrics
│   └── common.py      # Shared utilities
└── models/            # Trained models
    └── analyzer/      # Analyzer models
tests/                 # Test suite
├── test_agents/      # Agent-specific tests
├── test_integration/ # Integration tests
└── test_utils/      # Utility tests
logs/                 # Log files
config.yaml          # Configuration file
requirements.txt     # Dependencies
```

## Testing

The comprehensive test suite includes:

### Component Tests
- Base Agent (8 tests)
- Data Analyzer (8 tests)
- Web Scraper (15 tests)
- Basic Functionality (5 tests)
- Configuration (3 tests)

### Integration Tests
- End-to-end workflow tests
- Component interaction tests
- Error handling tests

Run tests with:
```bash
python -m pytest tests/ -v  # All tests
python -m pytest tests/test_agents/test_data_analyzer.py -v  # Specific component
```

Current test coverage: 42 tests, 100% pass rate

## Performance Metrics

The system includes a comprehensive metrics framework:

### Basic Metrics
- Processing time and throughput
- Success/error rates
- Risk assessment accuracy

### Advanced Metrics
- Discovery precision/recall
- Sensitivity classification accuracy
- Risk correlation analysis
- Recommendation relevance

### Visualization
- Metrics trends over time
- Correlation matrices
- Performance comparisons

## Future Work

Planned improvements:
- Real-time monitoring capabilities
- Multi-language support
- Web interface for visualization
- Enhanced ML model training
- Additional data type support
- Distributed processing support