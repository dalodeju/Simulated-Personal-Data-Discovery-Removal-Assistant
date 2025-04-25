"""Personal Data Discovery & Removal Assistant
Main script that coordinates all the agents and processes data.
"""

import logging
from datetime import datetime
import json
import os
from pathlib import Path

# Import our agents
from agents.web_scraper import WebScraperAgent
from agents.analyzer import DataAnalyzerAgent
from agents.risk import RiskEvaluator
from agents.recommender import RecommendationAgent
from utils.metrics import calculate_metrics, format_metrics_report

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def process_data(text_data):
    """Run text through all our agents and collect results"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'stages': {}
    }
    
    try:
        # Initialize all our agents
        scraper = WebScraperAgent()
        analyzer = DataAnalyzerAgent()
        risk_eval = RiskEvaluator()
        recommender = RecommendationAgent()
        
        # Step 1: Find personal data
        logging.info("Starting personal data discovery...")
        discoveries = scraper.discover(text_data)
        results['stages']['discovery'] = discoveries
        
        # Step 2: Analyze what we found
        logging.info("Analyzing discovered data...")
        analysis = analyzer.analyze(discoveries)
        results['stages']['analysis'] = analysis
        
        # Step 3: Evaluate risks
        logging.info("Evaluating risks...")
        risk_assessment = risk_eval.evaluate(analysis)
        results['stages']['risk'] = risk_assessment
        
        # Step 4: Generate recommendations
        logging.info("Generating recommendations...")
        recommendations = recommender.generate(risk_assessment)
        results['stages']['recommendations'] = recommendations
        
        return results
        
    except Exception as e:
        logging.error(f"Error processing data: {str(e)}")
        return {
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def save_results(results):
    """Save processing results to a file"""
    # Make sure we have a results directory
    results_dir = Path('experiments/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save to a timestamped file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f'scan_results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    return output_file

def main():
    """Main execution function"""
    setup_logging()
    logging.info("Starting personal data scan...")
    
    # Sample text with personal data (you'd normally get this from files/input)
    sample_text = """
    Please contact John at john.doe@email.com or call 555-123-4567.
    His SSN is 123-45-6789 and credit card: 4111-1111-1111-1111.
    """
    
    # Process the text
    results = process_data(sample_text)
    
    if 'error' in results:
        logging.error(f"Processing failed: {results['error']}")
        return
        
    # Calculate performance metrics
    metrics = calculate_metrics(results)
    
    # Save everything
    output_file = save_results(results)
    
    # Show summary
    print("\nProcessing Complete!")
    print(f"Found {len(results['stages']['discovery'])} items of personal data")
    print(f"Risk Level: {results['stages']['risk']['level']}")
    print(f"Generated {results['stages']['recommendations']['summary']['total_recommendations']} recommendations")
    print(f"\nResults saved to: {output_file}")
    print("\nPerformance Metrics:")
    print(format_metrics_report(metrics))

if __name__ == '__main__':
    main() 