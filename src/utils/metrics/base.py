"""Performance tracking for the personal data system.
Keeps track of how well we're doing at finding and handling personal data.
"""

from typing import Dict, Any
from datetime import datetime

def calculate_metrics(results):
    """Figure out how well the system is performing
    
    Takes the results from all processing stages and calculates
    various performance metrics like timing and success rates.
    """
    # Setup our metrics structure
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'timing': {},
        'performance': {},
        'summary': {}
    }
    
    try:
        # How long did it take?
        start = datetime.fromisoformat(results['timestamp'])
        end = datetime.now()
        processing_time = (end - start).total_seconds()
        
        # Get timing stats
        discoveries = results['stages']['discovery']
        metrics['timing'] = {
            'total_processing_time': processing_time,
            'processing_rate': (
                len(discoveries) / processing_time if processing_time > 0 
                else 0
            )
        }
        
        # How many things did we find?
        found_items = len(discoveries)
        risky_items = sum(
            1 for item in results['stages']['analysis'].get('items', [])
            if item.get('sensitivity') == 'high'
        )
        
        # Calculate performance stats
        metrics['performance'] = {
            'items_discovered': found_items,
            'high_risk_items': risky_items,
            'high_risk_ratio': (
                risky_items / found_items if found_items > 0 
                else 0
            )
        }
        
        # Get the final results
        metrics['summary'] = {
            'risk_level': results['stages']['risk']['level'],
            'recommendations_count': len(
                results['stages']['recommendations'].get('priority_actions', []) +
                results['stages']['recommendations'].get('protection_measures', [])
            )
        }
        
        return metrics
        
    except Exception as e:
        # Something went wrong - return error state
        return {
            'error': f"Couldn't calculate metrics: {str(e)}",
            'timestamp': datetime.now().isoformat()
        }

def format_metrics_report(metrics):
    """Make the metrics look nice for humans"""
    # Handle errors gracefully
    if 'error' in metrics:
        return f"Error calculating metrics: {metrics['error']}"
        
    # Build a nice report
    report = [
        "Performance Metrics Report",
        "========================",
        "",
        "Timing Metrics:",
        f"- Total Processing Time: {metrics['timing']['total_processing_time']:.2f} seconds",
        f"- Processing Rate: {metrics['timing']['processing_rate']:.2f} items/second",
        "",
        "Performance Metrics:",
        f"- Items Discovered: {metrics['performance']['items_discovered']}",
        f"- High Risk Items: {metrics['performance']['high_risk_items']}",
        f"- High Risk Ratio: {metrics['performance']['high_risk_ratio']:.2%}",
        "",
        "Summary:",
        f"- Risk Level: {metrics['summary']['risk_level']}",
        f"- Total Recommendations: {metrics['summary']['recommendations_count']}"
    ]
    
    return "\n".join(report) 