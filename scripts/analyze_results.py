#!/usr/bin/env python3
"""
Analyze ARLM experiment results.

Reads JSONL result files and computes accuracy metrics, confidence intervals,
average tokens, and average time. Optionally generates scaling curve plots.

Usage:
    # Single file
    python scripts/analyze_results.py results/phase1_baseline_100.jsonl
    
    # Compare multiple
    python scripts/analyze_results.py results/standard_r2.jsonl results/arlm_summary_r4.jsonl results/arlm_summary_r8.jsonl
    
    # Generate plot
    python scripts/analyze_results.py --plot results/scaling_curve.png results/*.jsonl
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from scipy.stats import beta


def clopper_pearson_ci(successes: int, trials: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Calculate Clopper-Pearson (exact) confidence interval for a binomial proportion.
    
    Args:
        successes: Number of successes
        trials: Total number of trials
        alpha: Significance level (default 0.05 for 95% CI)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    if trials == 0:
        return (0.0, 0.0)
    
    if successes == 0:
        lower = 0.0
    else:
        lower = beta.ppf(alpha / 2, successes, trials - successes + 1)
    
    if successes == trials:
        upper = 1.0
    else:
        upper = beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    
    return (lower, upper)


def load_results(filepath: str) -> List[Dict[str, Any]]:
    """Load results from a JSONL file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    results = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping malformed line in {filepath}: {e}", file=sys.stderr)
    
    return results


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate metrics from results.
    
    Expected fields in each result:
        - correct: bool
        - token_count: int (optional)
        - wall_time: float (optional)
    """
    n = len(results)
    if n == 0:
        return {
            'n': 0,
            'accuracy': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'avg_tokens': 0.0,
            'avg_time': 0.0,
        }
    
    correct_count = sum(1 for r in results if r.get('correct', False))
    accuracy = correct_count / n
    ci_lower, ci_upper = clopper_pearson_ci(correct_count, n)
    
    # Token count and time (may be missing in some results)
    token_counts = [r.get('token_count', 0) for r in results if 'token_count' in r]
    wall_times = [r.get('wall_time', 0) for r in results if 'wall_time' in r]
    
    avg_tokens = np.mean(token_counts) if token_counts else 0.0
    avg_time = np.mean(wall_times) if wall_times else 0.0
    
    return {
        'n': n,
        'accuracy': accuracy,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'avg_tokens': avg_tokens,
        'avg_time': avg_time,
    }


def extract_config_name(filepath: str) -> str:
    """Extract a human-readable config name from the filepath."""
    name = Path(filepath).stem
    return name


def parse_config_metadata(config_name: str) -> Dict[str, Any]:
    """
    Parse metadata from config name.
    
    Tries to extract:
        - mode: standard, arlm_summary, arlm_rlm, etc.
        - rounds: number of debate rounds (if present)
    
    Examples:
        standard_r2 -> mode=standard, rounds=2
        arlm_summary_r4 -> mode=arlm_summary, rounds=4
        phase1_baseline_20 -> mode=baseline, rounds=None
    """
    parts = config_name.split('_')
    metadata = {'mode': config_name, 'rounds': None}
    
    # Look for round indicator
    for part in parts:
        if part.startswith('r') and len(part) > 1 and part[1:].isdigit():
            metadata['rounds'] = int(part[1:])
    
    # Infer mode
    if 'arlm' in config_name.lower():
        if 'summary' in config_name.lower():
            metadata['mode'] = 'arlm_summary'
        elif 'rlm' in config_name.lower():
            metadata['mode'] = 'arlm_rlm'
        else:
            metadata['mode'] = 'arlm'
    elif 'standard' in config_name.lower() or 'debate' in config_name.lower():
        metadata['mode'] = 'standard'
    elif 'rlm' in config_name.lower():
        metadata['mode'] = 'rlm'
    elif 'single' in config_name.lower():
        metadata['mode'] = 'single'
    else:
        metadata['mode'] = 'baseline'
    
    return metadata


def format_results_table(file_metrics: List[Tuple[str, Dict[str, Any]]]) -> str:
    """Format results as a markdown table."""
    if not file_metrics:
        return "No results to display."
    
    lines = []
    lines.append("| Config | N | Accuracy | 95% CI | Avg Tokens | Avg Time |")
    lines.append("|--------|---|----------|--------|------------|----------|")
    
    for config_name, metrics in file_metrics:
        accuracy_pct = metrics['accuracy'] * 100
        ci_lower_pct = metrics['ci_lower'] * 100
        ci_upper_pct = metrics['ci_upper'] * 100
        ci_str = f"[{ci_lower_pct:.1f}, {ci_upper_pct:.1f}]"
        
        avg_tokens = int(metrics['avg_tokens'])
        avg_time_sec = metrics['avg_time']
        
        lines.append(
            f"| {config_name} | {metrics['n']} | {accuracy_pct:.1f}% | {ci_str} | {avg_tokens} | {avg_time_sec:.1f}s |"
        )
    
    return '\n'.join(lines)


def generate_plot(file_metrics: List[Tuple[str, Dict[str, Any]]], output_path: str):
    """
    Generate a scaling curve plot (accuracy vs rounds).
    
    Separate lines for different modes (standard, arlm_summary, arlm_rlm).
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required for plotting. Install with: pip install matplotlib", file=sys.stderr)
        sys.exit(1)
    
    # Group by mode
    mode_data = {}
    for config_name, metrics in file_metrics:
        metadata = parse_config_metadata(config_name)
        mode = metadata['mode']
        rounds = metadata.get('rounds')
        
        if rounds is None:
            # Try to infer from config name or skip
            if 'single' in config_name.lower() or 'r1' in config_name.lower():
                rounds = 1
            elif 'r2' in config_name.lower():
                rounds = 2
            else:
                continue  # Skip if we can't determine rounds
        
        if mode not in mode_data:
            mode_data[mode] = []
        
        mode_data[mode].append((rounds, metrics['accuracy'], metrics['ci_lower'], metrics['ci_upper']))
    
    # Sort by rounds for each mode
    for mode in mode_data:
        mode_data[mode].sort(key=lambda x: x[0])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {
        'standard': 'blue',
        'arlm_summary': 'green',
        'arlm_rlm': 'red',
        'arlm': 'purple',
        'single': 'gray',
        'baseline': 'orange',
        'rlm': 'brown',
    }
    
    for mode, data in mode_data.items():
        if not data:
            continue
        
        rounds = [d[0] for d in data]
        accuracies = [d[1] * 100 for d in data]  # Convert to percentage
        ci_lower = [d[2] * 100 for d in data]
        ci_upper = [d[3] * 100 for d in data]
        
        color = colors.get(mode, 'black')
        
        # Plot line
        ax.plot(rounds, accuracies, marker='o', label=mode, color=color, linewidth=2)
        
        # Add confidence interval shading
        ax.fill_between(rounds, ci_lower, ci_upper, color=color, alpha=0.2)
    
    ax.set_xlabel('Number of Rounds', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('ARLM Scaling Curve: Accuracy vs Debate Rounds', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze ARLM experiment results from JSONL files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze a single file
    python scripts/analyze_results.py results/phase1_baseline_100.jsonl
    
    # Compare multiple experiments
    python scripts/analyze_results.py results/phase1_single_100.jsonl results/phase1_debate_100.jsonl
    
    # Generate a scaling curve plot
    python scripts/analyze_results.py --plot results/scaling_curve.png results/phase2/*.jsonl
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='JSONL result files to analyze'
    )
    
    parser.add_argument(
        '--plot',
        type=str,
        default=None,
        help='Output path for scaling curve plot (e.g., results/scaling_curve.png)'
    )
    
    args = parser.parse_args()
    
    # Load and analyze each file
    file_metrics = []
    for filepath in args.files:
        try:
            results = load_results(filepath)
            metrics = compute_metrics(results)
            config_name = extract_config_name(filepath)
            file_metrics.append((config_name, metrics))
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Error processing {filepath}: {e}", file=sys.stderr)
            continue
    
    if not file_metrics:
        print("No valid result files to analyze.", file=sys.stderr)
        sys.exit(1)
    
    # Print results table
    print(format_results_table(file_metrics))
    print()
    
    # Generate plot if requested
    if args.plot:
        generate_plot(file_metrics, args.plot)


if __name__ == '__main__':
    main()
