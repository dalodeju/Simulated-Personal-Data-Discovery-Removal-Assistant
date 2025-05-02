"""Personal Data Discovery & Removal Assistant
Main script that coordinates all the agents and processes data.
"""

import logging  # configure logging for diagnostics
from datetime import datetime  # timestamping
import json  # json serialization
import os  # file system ops
from pathlib import Path  # path handling

# Import our agents
from agents.web_scraper import WebScraperAgent, DigitalEcosystem, ScraperConfig  # web scraper agent
from agents.data_analyzer import DataAnalyzerAgent, AnalyzerConfig  # nlp-based data analyzer
from agents.risk_evaluator import RiskEvaluatorAgent, RiskEvaluatorConfig  # risk scoring agent
from agents.recommender import RecommendationAgent  # generates recommendations
from utils.metrics import calculate_metrics, format_metrics_report  # metrics utilities
from environment.ecosystem import DigitalEcosystem, EcosystemConfig  # environment simulation

def setup_logging():
    """Setup basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )  # configure log format and level

def process_data(text_data):
    """Run text through all our agents and collect results"""
    results = {
        'timestamp': datetime.now().isoformat(),  # record execution time
        'stages': {}
    }
    
    try:
        # Initialize all our agents
        ecosystem = DigitalEcosystem(EcosystemConfig())  # setup the environment
        scraper = WebScraperAgent(ecosystem, ScraperConfig())  # instantiate web scraper
        analyzer = DataAnalyzerAgent(AnalyzerConfig())  # instantiate analyzer
        risk_eval = RiskEvaluatorAgent(RiskEvaluatorConfig())  # instantiate risk evaluator
        recommender = RecommendationAgent()  # instantiate recommender
        
        # Step 1: Find personal data
        logging.info("Starting personal data discovery...")
        discoveries = scraper.discover_profile_data("john")  # extract personal data
        results['stages']['discovery'] = discoveries
        
        # Step 2: Analyze what we found
        logging.info("Analyzing discovered data...")
        analysis = [analyzer.analyze_content({'id': d.get('profile_id'), 'content': str(d)}) for d in discoveries]  # classify and tag data
        results['stages']['analysis'] = analysis
        
        # Step 3: Evaluate risks
        logging.info("Evaluating risks...")
        risk_assessment = [risk_eval.evaluate_risk(a) for a in analysis]  # compute risk scores
        results['stages']['risk'] = risk_assessment
        
        # Step 4: Generate recommendations
        logging.info("Generating recommendations...")
        recommendations = [recommender.generate(r) for r in risk_assessment]  # generate mitigation actions
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
    results_dir = Path('experiments/results')  # output directory
    results_dir.mkdir(parents=True, exist_ok=True)  # ensure directory exists
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # unique filename
    output_file = results_dir / f'scan_results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)  # write results as json
        
    return output_file

def print_detailed_results(results, sample_size=5):
    discoveries = results['stages']['discovery']
    analysis = results['stages']['analysis']
    risks = results['stages']['risk']
    recommendations = results['stages']['recommendations']

    print("\n=== Detailed Results (Sample) ===")
    for i, (disc, ana, risk, rec) in enumerate(zip(discoveries, analysis, risks, recommendations)):
        if sample_size is not None and i >= sample_size:
            print(f"... ({len(discoveries) - sample_size} more not shown)")
            break
        print(f"\n--- Profile {i+1} ---")
        print(f"Profile ID: {disc.get('profile_id')}")
        print(f"Discovered Data: {json.dumps(disc.get('personal_data', {}), indent=2)}")
        print(f"Risk Level: {risk.get('level', 'unknown')}")
        print(f"Risk Score: {risk.get('risk_score', 0.0):.2f}")
        print("Actions/Recommendations:")
        for action in rec.get('priority_actions', []):
            print(f"  - {action}")
        for measure in rec.get('protection_measures', []):
            print(f"  - {measure}")

def compute_precision_recall(discoveries, ground_truths):
    """Compute precision, recall, and F1 for discovered vs. ground truth data."""
    tp, fp, fn = 0, 0, 0  # true/false positives/negatives
    for found, truth in zip(discoveries, ground_truths):
        found_set = set(found.get('personal_data', {}).keys())  # detected fields
        truth_set = set(truth.get('ground_truth', []))  # reference fields
        tp += len(found_set & truth_set)  # correct detections
        fp += len(found_set - truth_set)  # false positives
        fn += len(truth_set - found_set)  # false negatives
    precision = tp / (tp + fp) if (tp + fp) else 0  # precision metric
    recall = tp / (tp + fn) if (tp + fn) else 0  # recall metric
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0  # f1 score
    return precision, recall, f1

def run_pipeline(strategy='sequential', noise=0.1, risk_config=None, verbose=False):
    """Run the pipeline with specified strategy and noise, return metrics for experiments."""
    setup_logging()  # initialize logging
    ecosystem = DigitalEcosystem(EcosystemConfig(error_rate=noise))  # environment with noise
    scraper = WebScraperAgent(ecosystem, ScraperConfig())  # web scraper agent
    analyzer = DataAnalyzerAgent(AnalyzerConfig())  # analyzer agent
    risk_eval = RiskEvaluatorAgent(risk_config or RiskEvaluatorConfig())  # risk evaluator agent
    recommender = RecommendationAgent()  # recommendation agent

    # Coordination strategies
    if strategy == 'sequential':
        discoveries = scraper.discover_profile_data("john")  # sequential execution
        analysis = [analyzer.analyze_content({'id': d.get('profile_id'), 'content': str(d)}) for d in discoveries]
        risk_assessment = [risk_eval.evaluate_risk(a) for a in analysis]
        recommendations = [recommender.generate(r) for r in risk_assessment]
    elif strategy == 'parallel':
        discoveries = scraper.discover_profile_data("john")  # parallel execution (simulated)
        analysis = [analyzer.analyze_content({'id': d.get('profile_id'), 'content': str(d)}) for d in discoveries]
        risk_assessment = [risk_eval.evaluate_risk(a) for a in analysis]
        recommendations = [recommender.generate(r) for r in risk_assessment]
    elif strategy == 'hybrid':
        discoveries = scraper.discover_profile_data("john")
        analysis = []
        risk_assessment = []
        recommendations = []
        for i, d in enumerate(discoveries):
            if i % 2 == 0:
                a = analyzer.analyze_content({'id': d.get('profile_id'), 'content': str(d)})
                r = risk_eval.evaluate_risk(a)
                rec = recommender.generate(r)
            else:
                a = analyzer.analyze_content({'id': d.get('profile_id'), 'content': str(d)})
                r = risk_eval.evaluate_risk(a)
                rec = recommender.generate(r)
            analysis.append(a)
            risk_assessment.append(r)
            recommendations.append(rec)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")  # invalid strategy guard

    # Metrics
    discovered_ids = [d.get('profile_id') for d in discoveries]  # discovered profile ids
    ground_truths = [next((p for p in ecosystem.profiles if p['id'] == str(pid)), {}) for pid in discovered_ids]  # reference profiles
    precision, recall, f1 = compute_precision_recall(discoveries, ground_truths)  # compute metrics
    metrics = calculate_metrics({
        'timestamp': datetime.now().isoformat(),
        'stages': {
            'discovery': discoveries,
            'analysis': analysis,
            'risk': risk_assessment,
            'recommendations': recommendations
        }
    })
    if verbose:
        print(f"Strategy: {strategy}, Noise: {noise}")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        print(format_metrics_report(metrics))
    return {
        'strategy': strategy,
        'noise': noise,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'processing_time': metrics['timing']['total_processing_time'],
        'risk_level': metrics['summary']['risk_level'],
        'risk_level_counts': metrics['summary']['risk_level_counts'],
        'recommendations_count': metrics['summary']['recommendations_count']
    }

def main():
    """Main execution function"""
    setup_logging()  # initialize logging
    logging.info("Starting personal data scan...")
    
    # Sample text with personal data (you'd normally get this from files/input)
    sample_text = """
    Please contact John at john.doe@email.com or call 555-123-4567.
    His SSN is 123-45-6789 and credit card: 4111-1111-1111-1111.
    """
    
    # Process the text
    results = process_data(sample_text)  # run pipeline
    
    if 'error' in results:
        logging.error(f"Processing failed: {results['error']}")
        return
        
    # Calculate performance metrics
    metrics = calculate_metrics(results)  # aggregate metrics
    
    # Save everything
    output_file = save_results(results)  # persist results
    
    # Compute precision, recall, F1
    ecosystem = DigitalEcosystem(EcosystemConfig())  # reload environment for ground truth
    discovered_ids = [d.get('profile_id') for d in results['stages']['discovery']]
    ground_truths = [next((p for p in ecosystem.profiles if p['id'] == str(pid)), {}) for pid in discovered_ids]
    precision, recall, f1 = compute_precision_recall(results['stages']['discovery'], ground_truths)
    
    # Show summary
    print("\nProcessing Complete!")
    print(f"Found {len(results['stages']['discovery'])} items of personal data")
    print(f"Risk Levels: {[r.get('level', 'unknown') for r in results['stages']['risk']]}")
    print(f"Generated {sum(r['summary']['total_recommendations'] for r in results['stages']['recommendations'])} recommendations")
    print(f"\nResults saved to: {output_file}")
    print("\nPerformance Metrics:")
    print(format_metrics_report(metrics))
    print(f"\nPrecision: {precision:.2f}  Recall: {recall:.2f}  F1: {f1:.2f}")
    print_detailed_results(results, sample_size=5)

if __name__ == '__main__':
    main()  # entry point 