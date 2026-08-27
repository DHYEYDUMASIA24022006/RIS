#!/usr/bin/env python3
"""Three-Way Comparison: M1 vs M2 vs M3.

Loads results from all three experiments and generates:
  1. Unified comparison table (CSV + JSON)
  2. Combined convergence overlay plot
  3. Metric comparison bar charts
  4. Summary statistics for the paper

Usage:
    python analysis/compare_methods.py
    python analysis/compare_methods.py --results-dir results
    python analysis/compare_methods.py --results-dir results/multi_seed/seed_42
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_experiment_results(results_dir: str) -> dict:
    """Load summary JSONs from each experiment directory."""
    data = {}

    # Exp1 summaries
    exp1_dir = os.path.join(results_dir, 'exp1')
    if os.path.isdir(exp1_dir):
        data['exp1'] = {}
        for jf in sorted(glob.glob(os.path.join(exp1_dir, '*.json'))):
            with open(jf) as f:
                name = os.path.basename(jf).replace('.json', '')
                data['exp1'][name] = json.load(f)

    # Exp2 summary
    exp2_summary = os.path.join(results_dir, 'exp2', 'exp2_summary.json')
    if os.path.exists(exp2_summary):
        with open(exp2_summary) as f:
            data['exp2'] = json.load(f)

    # Exp3 summary
    exp3_summary = os.path.join(results_dir, 'exp3', 'exp3_summary.json')
    if os.path.exists(exp3_summary):
        with open(exp3_summary) as f:
            data['exp3'] = json.load(f)

    return data


def load_csv_metrics(results_dir: str) -> dict:
    """Load per-round CSV metrics for convergence overlay."""
    csvs = {}
    for exp_dir in ['exp1', 'exp2', 'exp3']:
        full_dir = os.path.join(results_dir, exp_dir)
        if not os.path.isdir(full_dir):
            continue
        csvs[exp_dir] = {}
        for csv_file in sorted(glob.glob(os.path.join(full_dir, '*.csv'))):
            name = os.path.basename(csv_file).replace('.csv', '')
            try:
                df = pd.read_csv(csv_file)
                csvs[exp_dir][name] = df
            except Exception:
                pass
    return csvs


def generate_three_way_table(data: dict, save_dir: str):
    """Generate unified comparison table across M1, M2, M3."""
    rows = []

    # M1: best condition is Fading + Optimised RIS
    if 'exp1' in data:
        for name, summary in data['exp1'].items():
            if isinstance(summary, dict) and 'best_test_accuracy' in summary:
                rows.append({
                    'Methodology': 'M1 (FL +/- RIS)',
                    'Condition': summary.get('experiment', name),
                    'Final Accuracy (%)': summary.get('final_test_accuracy', 0),
                    'Best Accuracy (%)': summary.get('best_test_accuracy', 0),
                    'Mean SNR (dB)': summary.get('mean_snr_db', 0),
                    'Outage Prob': summary.get('mean_outage_probability', 0),
                    'Agg MSE': summary.get('mean_aggregation_mse', 0),
                    'Mean Latency (ms)': summary.get('mean_latency_ms', 0),
                    'Mean Energy (mJ)': summary.get('mean_energy_mj', 0),
                    'Fairness (Jain)': summary.get('final_successful_upload_fairness', 0),
                    'Rounds to Target': summary.get('rounds_to_target', -1),
                })

    # M2: convergence results
    if 'exp2' in data and 'convergence' in data['exp2']:
        for strat_name, strat_data in data['exp2']['convergence'].items():
            rows.append({
                'Methodology': 'M2 (Joint Selection)',
                'Condition': strat_name,
                'Final Accuracy (%)': strat_data.get('final_accuracy_percent', 0),
                'Best Accuracy (%)': strat_data.get('final_accuracy_percent', 0),
                'Mean SNR (dB)': strat_data.get('mean_snr_db', 0),
                'Outage Prob': strat_data.get('mean_outage_probability', 0),
                'Agg MSE': strat_data.get('mean_aggregation_mse', 0),
                'Mean Latency (ms)': strat_data.get('mean_latency_ms', 0),
                'Mean Energy (mJ)': strat_data.get('mean_energy_mj', 0),
                'Fairness (Jain)': strat_data.get('successful_upload_fairness', 0),
                'Rounds to Target': -1,
            })

    # M3: final accuracy across alpha values
    if 'exp3' in data and 'final_accuracy_percent' in data['exp3']:
        for method, alpha_data in data['exp3']['final_accuracy_percent'].items():
            for alpha, acc in alpha_data.items():
                rows.append({
                    'Methodology': 'M3 (AirComp)',
                    'Condition': f'{method} (alpha={alpha})',
                    'Final Accuracy (%)': acc,
                    'Best Accuracy (%)': acc,
                    'Mean SNR (dB)': data['exp3'].get('mean_snr_db', {}).get(method, {}).get(alpha, 0),
                    'Outage Prob': data['exp3'].get('mean_outage_probability', {}).get(method, {}).get(alpha, 0),
                    'Agg MSE': data['exp3'].get('mean_aggregation_mse', {}).get(method, {}).get(alpha, 0),
                    'Mean Latency (ms)': data['exp3'].get('mean_latency_ms', {}).get(method, {}).get(alpha, 0),
                    'Mean Energy (mJ)': data['exp3'].get('mean_energy_mj', {}).get(method, {}).get(alpha, 0),
                    'Fairness (Jain)': data['exp3'].get(
                        'successful_upload_fairness', {}).get(
                        method, {}).get(alpha, 0),
                    'Rounds to Target': -1,
                })

    if not rows:
        print('No experiment results found to compare.')
        return None

    df = pd.DataFrame(rows)
    csv_path = os.path.join(save_dir, 'three_way_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.4f')
    print(f'  Three-way comparison table saved to: {csv_path}')

    json_path = os.path.join(save_dir, 'three_way_comparison.json')
    with open(json_path, 'w') as f:
        json.dump(rows, f, indent=2)

    # Print
    print(f'\n{"="*120}')
    print(f'  THREE-WAY METHODOLOGY COMPARISON (M1 vs M2 vs M3)')
    print(f'{"="*120}')
    print(df.to_string(index=False, max_colwidth=35))
    print(f'{"="*120}')

    return df


def generate_combined_plots(csv_data: dict, save_dir: str,
                            dpi: int = 300, fmt: str = 'pdf'):
    """Generate combined comparison plots."""
    sns.set_style('whitegrid')
    fontsize = 14
    plt.rcParams.update({'font.size': fontsize})

    # 1. Best-of-each-methodology convergence overlay
    best_curves = {}

    # M1 best: Fading + Optimised RIS
    if 'exp1' in csv_data:
        for name, df in csv_data['exp1'].items():
            if 'Optimised_RIS' in name and 'test_accuracy' in df.columns:
                best_curves['M1: Fading + Optimised RIS'] = df
            elif 'No_Fading' in name and 'test_accuracy' in df.columns:
                best_curves['M1: Ideal (Upper Bound)'] = df

    # M2 best: Joint+RIS
    if 'exp2' in csv_data:
        for name, df in csv_data['exp2'].items():
            if 'joint' in name.lower() and 'test_accuracy' in df.columns:
                best_curves['M2: Joint+RIS (Proposed)'] = df
                break

    # M3 best: AirComp+RIS
    if 'exp3' in csv_data:
        for name, df in csv_data['exp3'].items():
            if 'aircomp' in name.lower() and 'ris' in name.lower() and 'test_accuracy' in df.columns:
                best_curves['M3: AirComp + RIS'] = df
                break

    if best_curves:
        colors = sns.color_palette('Set1', len(best_curves))
        markers = ['o', 's', '^', 'D', 'v']

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (label, df) in enumerate(best_curves.items()):
            rounds = df['round'].values if 'round' in df.columns else range(len(df))
            acc = df['test_accuracy'].values
            ax.plot(rounds, acc, label=label, color=colors[i],
                    linewidth=2.5, marker=markers[i % len(markers)],
                    markevery=max(1, len(rounds)//10), markersize=7)

        ax.set_xlabel('Communication Round')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Three-Way Comparison: Best of Each Methodology')
        ax.legend(loc='lower right', fontsize=fontsize - 2)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = os.path.join(save_dir, f'three_way_convergence.{fmt}')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'  Combined convergence plot saved: {path}')

    # 2. Metric comparison bar chart (best condition from each methodology)
    metric_labels = ['Final Accuracy (%)', 'Mean Latency (ms)',
                     'Mean Energy (mJ)', 'Fairness (Jain)']
    methodology_labels = []
    metric_values = {m: [] for m in metric_labels}

    # Collect best values from each experiment's JSON summaries
    _collect_best_metrics(csv_data, methodology_labels, metric_values)

    if methodology_labels:
        fig, axes = plt.subplots(1, len(metric_labels), figsize=(16, 5))
        colors = sns.color_palette('Set2', len(methodology_labels))

        for ax_idx, metric in enumerate(metric_labels):
            ax = axes[ax_idx] if len(metric_labels) > 1 else axes
            vals = metric_values[metric]
            if vals:
                bars = ax.bar(range(len(vals)), vals, color=colors[:len(vals)],
                              edgecolor='black', linewidth=0.5)
                ax.set_xticks(range(len(vals)))
                ax.set_xticklabels(methodology_labels, rotation=25,
                                   ha='right', fontsize=fontsize - 3)
                ax.set_ylabel(metric, fontsize=fontsize - 2)
                ax.grid(True, alpha=0.3, axis='y')
                for bar, val in zip(bars, vals):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{val:.1f}', ha='center', va='bottom',
                            fontsize=fontsize - 3)

        fig.suptitle('Methodology Comparison: Key Metrics', fontsize=fontsize + 2)
        fig.tight_layout()
        path = os.path.join(save_dir, f'metric_comparison.{fmt}')
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        print(f'  Metric comparison plot saved: {path}')


def _collect_best_metrics(csv_data: dict, labels: list, metrics: dict):
    """Collect best metric values from CSV data for bar chart."""
    for exp_name, exp_csvs in csv_data.items():
        for name, df in exp_csvs.items():
            if 'test_accuracy' not in df.columns:
                continue
            # Take last row as representative
            last = df.iloc[-1]
            short_name = name.replace('exp1_', 'M1: ').replace(
                'exp2_', 'M2: ').replace('exp3_', 'M3: ')
            if len(short_name) > 30:
                short_name = short_name[:27] + '...'

            labels.append(short_name)
            metrics['Final Accuracy (%)'].append(
                float(last.get('test_accuracy', 0)))
            metrics['Mean Latency (ms)'].append(
                float(last.get('per_round_latency_ms', 0)))
            metrics['Mean Energy (mJ)'].append(
                float(last.get('mean_client_energy_mj', 0)))
            metrics['Fairness (Jain)'].append(
                float(last.get('jain_fairness_index', 0)))


def generate_comparison(results_dir: str, dpi: int = 300, fmt: str = 'pdf'):
    """Load experiment outputs and write the three-way comparison artifacts."""
    save_dir = os.path.join(results_dir, 'comparison')
    os.makedirs(save_dir, exist_ok=True)

    print(f'Loading results from: {results_dir}')
    data = load_experiment_results(results_dir)
    csv_data = load_csv_metrics(results_dir)

    if not data and not csv_data:
        print('ERROR: No results found. Run experiments first:')
        print('  python main.py --experiment all')
        return None

    print('\nGenerating three-way comparison table...')
    df = generate_three_way_table(data, save_dir)

    print('\nGenerating combined plots...')
    generate_combined_plots(csv_data, save_dir, dpi=dpi, fmt=fmt)

    print(f'\nAll comparison outputs saved to: {save_dir}/')
    return df


def main():
    parser = argparse.ArgumentParser(
        description='Three-Way Comparison: M1 vs M2 vs M3')
    parser.add_argument('--results-dir', type=str, default='results',
                        help='Root results directory')
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--format', type=str, default='pdf',
                        choices=['pdf', 'png', 'eps'])
    args = parser.parse_args()

    df = generate_comparison(args.results_dir, dpi=args.dpi, fmt=args.format)
    if df is None:
        sys.exit(1)


if __name__ == '__main__':
    main()
