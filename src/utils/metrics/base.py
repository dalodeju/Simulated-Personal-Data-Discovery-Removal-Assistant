"""
performance tracking for the personal data system.
this module provides functions for calculating and formatting system performance metrics.
"""

from typing import Dict, Any
from datetime import datetime
from collections import Counter

def calculate_metrics(results):
    """
    calculate system performance metrics from experiment results.
    returns a dictionary with timing, performance, and summary statistics.
    """
    # set up our metrics structure
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'timing': {},
        'performance': {},
        'summary': {}
    }
    try:
        # how long did it take?
        start = datetime.fromisoformat(results['timestamp'])
        end = datetime.now()
        processing_time = (end - start).total_seconds()
        # get timing stats
        discoveries = results['stages']['discovery']
        metrics['timing'] = {
            'total_processing_time': processing_time,
            'processing_rate': (
                len(discoveries) / processing_time if processing_time > 0 
                else 0
            )
        }
        # how many things did we find?
        found_items = len(discoveries)
        # handle analysis as a list
        analysis = results['stages']['analysis']
        risky_items = sum(
            1 for item in analysis
            if item.get('sensitivity', {}).get('class') == 'sensitive'
        )
        # calculate performance stats
        metrics['performance'] = {
            'items_discovered': found_items,
            'high_risk_items': risky_items,
            'high_risk_ratio': (
                risky_items / found_items if found_items > 0 
                else 0
            )
        }
        # handle risk as a list
        risks = results['stages']['risk']
        if isinstance(risks, list):
            risk_levels = [r.get('level', 'unknown') for r in risks]
            # count each risk level
            level_counts = Counter(risk_levels)
            # most severe risk (critical > high > medium > low > unknown)
            level_order = ['low', 'medium', 'high', 'critical']
            def level_value(lvl):
                try:
                    return level_order.index(lvl)
                except ValueError:
                    return -1
            most_severe = max(risk_levels, key=level_value, default='unknown')
        else:
            risk_levels = [risks.get('level', 'unknown')]
            level_counts = Counter(risk_levels)
            most_severe = risk_levels[0]
        # handle recommendations as a list
        recs = results['stages']['recommendations']
        if isinstance(recs, list):
            total_recs = sum(
                len(r.get('priority_actions', [])) + len(r.get('protection_measures', []))
                for r in recs
            )
        else:
            total_recs = len(recs.get('priority_actions', [])) + len(recs.get('protection_measures', []))
        # get the final results
        metrics['summary'] = {
            'risk_level': most_severe,
            'risk_level_counts': dict(level_counts),
            'recommendations_count': total_recs
        }
        return metrics
    except Exception as e:
        # something went wrong - return error state
        return {
            'error': f"couldn't calculate metrics: {str(e)}",
            'timestamp': datetime.now().isoformat()
        }

def format_metrics_report(metrics):
    """
    format the metrics dictionary into a human-readable report string.
    """
    # handle errors gracefully
    if 'error' in metrics:
        return f"error calculating metrics: {metrics['error']}"
    # build a nice report
    report = [
        "performance metrics report",
        "========================",
        "",
        "timing metrics:",
        f"- total processing time: {metrics['timing']['total_processing_time']:.2f} seconds",
        f"- processing rate: {metrics['timing']['processing_rate']:.2f} items/second",
        "",
        "performance metrics:",
        f"- items discovered: {metrics['performance']['items_discovered']}",
        f"- high risk items: {metrics['performance']['high_risk_items']}",
        f"- high risk ratio: {metrics['performance']['high_risk_ratio']:.2%}",
        "",
        "summary:",
        f"- most severe risk level: {metrics['summary']['risk_level']}",
        f"- risk level counts: {metrics['summary']['risk_level_counts']}",
        f"- total recommendations: {metrics['summary']['recommendations_count']}"
    ]
    return "\n".join(report) 