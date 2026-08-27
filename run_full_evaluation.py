#!/usr/bin/env python3
"""RIS-Assisted Wireless Federated Learning — Full Multi-Seed Evaluation.

Runs all 3 experiments across multiple random seeds for statistical
significance (mean ± std).  Results are written to results/multi_seed/.

Usage:
    python run_full_evaluation.py                     # Default 3 seeds
    python run_full_evaluation.py --seeds 42 123 456  # Custom seeds
    python run_full_evaluation.py --fast               # 50-round MNIST quick run
"""
import argparse
import copy
import json
import os
import sys
import time
import numpy as np
import torch
import yaml

from experiments.exp1_baseline import run_experiment as run_exp1
from experiments.exp2_joint_design import run_experiment as run_exp2
from experiments.exp3_aircomp import run_experiment as run_exp3


def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _collect_json_summaries(results_dir: str) -> dict:
    """Walk *results_dir* and collect every JSON summary file."""
    summaries = {}
    for root, _dirs, files in os.walk(results_dir):
        for fname in sorted(files):
            if fname.endswith('.json'):
                path = os.path.join(root, fname)
                with open(path) as f:
                    summaries[fname] = json.load(f)
    return summaries


def _aggregate_across_seeds(all_seed_summaries: list[dict]) -> dict:
    """Compute mean ± std for numeric fields across seed runs.

    Parameters
    ----------
    all_seed_summaries : list[dict]
        Each element is the dict returned by ``_collect_json_summaries``
        for one seed.

    Returns
    -------
    dict
        ``{filename: {field: {"mean": ..., "std": ...}}}``
        for every numeric field found in any summary file.
    """
    # Collect all filenames
    all_keys: set[str] = set()
    for s in all_seed_summaries:
        all_keys.update(s.keys())

    aggregated: dict = {}
    for key in sorted(all_keys):
        values_per_field: dict[str, list[float]] = {}
        for seed_summary in all_seed_summaries:
            if key not in seed_summary:
                continue
            _flatten_numeric(seed_summary[key], '', values_per_field)

        agg: dict = {}
        for field, vals in values_per_field.items():
            if vals:
                agg[field] = {
                    'mean': float(np.mean(vals)),
                    'std': float(np.std(vals)),
                    'values': [float(v) for v in vals],
                }
        if agg:
            aggregated[key] = agg
    return aggregated


def _flatten_numeric(obj, prefix: str, out: dict):
    """Recursively extract numeric leaf values into *out*."""
    if isinstance(obj, (int, float)):
        out.setdefault(prefix, []).append(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = f'{prefix}.{k}' if prefix else k
            _flatten_numeric(v, new_prefix, out)
    # lists / strings / None are silently skipped


def run_multi_seed(config: dict, seeds: list[int], experiments: list[str],
                   base_results_dir: str = 'results'):
    """Run experiments across multiple seeds and aggregate."""

    multi_dir = os.path.join(base_results_dir, 'multi_seed')
    os.makedirs(multi_dir, exist_ok=True)

    all_seed_summaries = []

    for seed in seeds:
        print(f'\n{"#"*60}')
        print(f'#  SEED {seed}')
        print(f'{"#"*60}')

        set_seeds(seed)
        cfg = copy.deepcopy(config)
        cfg.setdefault('system', {})['seed'] = seed

        seed_dir = os.path.join(multi_dir, f'seed_{seed}')
        cfg['system']['results_dir'] = seed_dir
        os.makedirs(seed_dir, exist_ok=True)

        exp3_done = os.path.exists(os.path.join(seed_dir, 'exp3', 'exp3_summary.json'))
        if exp3_done:
            print(f'Seed {seed} already complete; skipping experiment rerun.')
        else:
            if 'baseline' in experiments or 'all' in experiments:
                print(f'\n--- Experiment 1 (seed={seed}) ---')
                run_exp1(cfg)

            if 'joint' in experiments or 'all' in experiments:
                print(f'\n--- Experiment 2 (seed={seed}) ---')
                run_exp2(cfg)

            if 'aircomp' in experiments or 'all' in experiments:
                print(f'\n--- Experiment 3 (seed={seed}) ---')
                run_exp3(cfg)

        summaries = _collect_json_summaries(seed_dir)
        all_seed_summaries.append(summaries)

        from analysis.compare_methods import generate_comparison
        generate_comparison(seed_dir)

    # Aggregate across seeds
    print(f'\n{"="*60}')
    print('Aggregating results across seeds...')
    aggregated = _aggregate_across_seeds(all_seed_summaries)

    agg_path = os.path.join(multi_dir, 'aggregated_results.json')
    with open(agg_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f'Aggregated results saved to: {agg_path}')

    # Print summary table
    _print_summary(aggregated, seeds)

    return aggregated


def _print_summary(aggregated: dict, seeds: list[int]):
    """Print a readable summary of the aggregated results."""
    print(f'\n{"="*80}')
    print(f'  MULTI-SEED SUMMARY ({len(seeds)} seeds: {seeds})')
    print(f'{"="*80}')

    for filename, metrics in aggregated.items():
        print(f'\n  {filename}:')
        for metric, stats in metrics.items():
            if 'accuracy' in metric.lower() or 'acc' in metric.lower():
                print(f'    {metric}: {stats["mean"]:.2f} +/- {stats["std"]:.2f}')
            elif 'mse' in metric.lower():
                print(f'    {metric}: {stats["mean"]:.2e} +/- {stats["std"]:.2e}')
            elif any(k in metric.lower() for k in ['snr', 'outage', 'fairness', 'latency', 'energy']):
                print(f'    {metric}: {stats["mean"]:.4f} +/- {stats["std"]:.4f}')

    print(f'\n{"="*80}')


def main():
    parser = argparse.ArgumentParser(
        description='Multi-seed full evaluation for RIS-Assisted Wireless FL')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds to evaluate')
    parser.add_argument('--experiments', type=str, nargs='+',
                        default=['all'],
                        choices=['all', 'baseline', 'joint', 'aircomp'],
                        help='Which experiments to run')
    parser.add_argument('--fast', action='store_true',
                        help='Use reduced rounds and MNIST for quick evaluation')
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f'Config not found: {args.config}, using defaults.')
        config = {}

    if args.fast:
        config.setdefault('fl', {})
        config['fl']['num_rounds'] = 6
        config['fl']['num_clients'] = 6
        config['fl']['local_epochs'] = 1
        config['fl']['batch_size'] = 64
        config.setdefault('data', {})
        config['data']['dataset'] = 'mnist'
        config['data']['partition'] = 'iid'
        config['data']['max_samples_per_client'] = 192
        config.setdefault('channel', {})
        config['channel']['num_ris_elements'] = 16
        config.setdefault('ris', {})
        config['ris']['max_iterations'] = 8
        config.setdefault('experiments', {})
        config['experiments']['exp1_rounds'] = 6
        config['experiments']['exp3_fl_rounds'] = 5
        config['experiments']['exp2_skip_sweeps'] = True
        config['experiments']['exp3_mse_trials'] = 8
        config['experiments']['exp3_ris_elements_sweep'] = [16, 32, 64]
        config['experiments']['exp3_snr_sweep'] = [0, 10, 20]
        config['experiments']['exp3_alpha_sweep'] = [0.1, 0.5, 100.0]
        print('FAST MODE: MNIST, 6 clients, 6 rounds, skipped Exp2 sweeps')

    start = time.time()
    run_multi_seed(config, args.seeds, args.experiments)
    elapsed = time.time() - start

    print(f'\nTotal evaluation time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)')


if __name__ == '__main__':
    main()
