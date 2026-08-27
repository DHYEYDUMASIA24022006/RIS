#!/usr/bin/env python3
"""RIS-Assisted Wireless Federated Learning — Main Runner.

Usage:
    python main.py --experiment all         # Run all experiments
    python main.py --experiment baseline    # Run Experiment 1 only
    python main.py --experiment joint       # Run Experiment 2 only
    python main.py --experiment aircomp     # Run Experiment 3 only
    python main.py --experiment quick       # Quick smoke test (3 rounds, 4 clients, MNIST)
    python main.py --config custom.yaml     # Use custom config
"""
import argparse
import os
import sys
import time
import yaml
import numpy as np
import torch


def load_config(path: str) -> dict:
    """Load YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def setup_device(config: dict) -> torch.device:
    """Setup compute device from config."""
    dev = config.get('system', {}).get('device', 'auto')
    if dev == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(dev)


def set_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_quick_test(config: dict):
    """Stressed smoke test: MNIST, few rounds, blocked direct links.

    Kept short, but the channel must be able to drop uploads so RIS can matter.
    """
    config.setdefault('fl', {})
    config['fl']['num_clients'] = 4
    config['fl']['num_rounds'] = 3
    config['fl']['local_epochs'] = 1
    config['fl']['fraction_fit'] = 1.0
    config['fl']['snr_outage_threshold_db'] = 0.0
    config['fl']['update_noise_scale'] = 0.10
    config['fl']['use_rate_outage'] = False
    config['fl']['enforce_energy_latency'] = False

    config.setdefault('data', {})
    config['data']['dataset'] = 'mnist'
    config['data']['partition'] = 'iid'
    config['data']['max_samples_per_client'] = 128
    config['fl']['batch_size'] = 64

    config.setdefault('channel', {})
    config['channel']['direct_blockage_prob'] = 0.85
    config['channel']['direct_blockage_attenuation_db'] = 35.0

    config.setdefault('experiments', {})
    config['experiments']['exp1_rounds'] = 3

    config.setdefault('system', {})
    config['system']['results_dir'] = os.path.join(
        config['system'].get('results_dir', 'results'), 'quick')

    from experiments.exp1_baseline import run_experiment
    run_experiment(config)


def main():
    parser = argparse.ArgumentParser(
        description='RIS-Assisted Wireless Federated Learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'baseline', 'joint', 'aircomp', 'quick'],
                        help='Which experiment to run')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--seed', type=int, default=None,
                        help='Override random seed')
    args = parser.parse_args()

    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Warning: Config file {args.config} not found, using defaults.")
        config = {}

    if args.seed is not None:
        config.setdefault('system', {})['seed'] = args.seed

    seed = config.get('system', {}).get('seed', 42)
    set_seeds(seed)
    device = setup_device(config)

    print('=' * 60)
    print('  RIS-Assisted Wireless Federated Learning')
    print(f'  Device: {device}')
    print(f'  Experiment: {args.experiment}')
    print(f'  Seed: {seed}')
    print('=' * 60)

    results_dir = config.get('system', {}).get('results_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)

    start = time.time()

    if args.experiment in ['all', 'baseline']:
        from experiments.exp1_baseline import run_experiment as run_exp1
        run_exp1(config)

    if args.experiment in ['all', 'joint']:
        from experiments.exp2_joint_design import run_experiment as run_exp2
        run_exp2(config)

    if args.experiment in ['all', 'aircomp']:
        from experiments.exp3_aircomp import run_experiment as run_exp3
        run_exp3(config)

    if args.experiment == 'quick':
        run_quick_test(config)

    results_dir = config.get('system', {}).get('results_dir', 'results')

    if args.experiment == 'all':
        from analysis.compare_methods import generate_comparison
        generate_comparison(results_dir)

    elapsed = time.time() - start
    print(f'\n{"="*60}')
    print(f'  Total time: {elapsed/60:.1f} minutes ({elapsed:.0f} seconds)')
    print(f'  Results saved to: {os.path.abspath(results_dir)}/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()
