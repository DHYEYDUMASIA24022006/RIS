"""Experiment 3: AirComp with RIS + non-IID study.

Studies:
  (a) AirComp aggregation MSE vs N (RIS elements): [16, 32, 64, 128]
  (b) AirComp MSE vs SNR: [-5, 0, 5, 10, 15, 20] dB
  (c) FL accuracy under IID vs non-IID (Dirichlet alpha: 0.1, 0.5, 1.0, 100)
  (d) AirComp vs FedAvg accuracy comparison
  (e) Fairness (Jain's index) vs alpha

Usage:
    python experiments/exp3_aircomp.py
    python experiments/exp3_aircomp.py --config config/config.yaml
"""

import argparse
import copy
import json
import os
import sys
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel.channel_model import WirelessChannel
from channel.ris_optimizer import RISOptimizer
from fl.models import get_model
from fl.data_partition import (load_dataset, iid_partition,
                                dirichlet_noniid_partition,
                                get_client_dataloaders)
from fl.client import FLClient
from fl.server import FLServer
from fl.aircomp import AirCompAggregator
from metrics.metrics_logger import MetricsLogger
from utils.plotting import save_figure


def run_experiment(config: dict):
    """Run Experiment 3: AirComp + non-IID study."""

    seed = config.get('system', {}).get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fl_cfg = config.get('fl', {})
    ch_cfg = config.get('channel', {})
    data_cfg = config.get('data', {})
    exp_cfg = config.get('experiments', {})

    num_clients = fl_cfg.get('num_clients', 20)
    num_rounds = fl_cfg.get('num_rounds', 100)
    batch_size = fl_cfg.get('batch_size', 32)
    dataset_name = data_cfg.get('dataset', 'mnist')

    N_sweep = exp_cfg.get('exp3_ris_elements_sweep', [16, 32, 64, 128])
    snr_sweep = exp_cfg.get('exp3_snr_sweep', [-5, 0, 5, 10, 15, 20])
    alpha_sweep = exp_cfg.get('exp3_alpha_sweep', data_cfg.get('dirichlet_alpha', [0.1, 0.5, 1.0, 100.0]))

    save_dir = os.path.join(config.get('system', {}).get('results_dir', 'results'), 'exp3')
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment 3: AirComp + non-IID Study")
    print(f"Dataset: {dataset_name} | Clients: {num_clients}")
    print(f"N sweep: {N_sweep} | SNR sweep: {snr_sweep}")
    print(f"Alpha sweep: {alpha_sweep}")
    print(f"{'='*60}\n")

    # ---- Data ---- #
    train_dataset, test_dataset = load_dataset(dataset_name)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    def model_fn():
        return get_model(dataset_name, torch.device('cpu'))

    # ============================================================= #
    # Part 1: AirComp MSE vs N (RIS elements)                        #
    # ============================================================= #
    print("Part 1: AirComp MSE vs N (RIS elements)")
    mse_vs_n = _aircomp_mse_sweep(N_sweep, num_clients, config, sweep_type='N',
                                 num_trials=int(exp_cfg.get('exp3_mse_trials', 50)))

    # ============================================================= #
    # Part 2: AirComp MSE vs SNR                                      #
    # ============================================================= #
    print("\nPart 2: AirComp MSE vs SNR")
    mse_vs_snr = _aircomp_mse_sweep(snr_sweep, num_clients, config, sweep_type='SNR',
                                   num_trials=int(exp_cfg.get('exp3_mse_trials', 50)))

    # ============================================================= #
    # Part 3: FL accuracy with different alpha values                  #
    # ============================================================= #
    print("\nPart 3: FL accuracy across data heterogeneity levels")

    methods = {
        'FedAvg (Ideal)':  {'uses_channel': False, 'use_ris': False, 'aircomp': False},
        'FedAvg + RIS':    {'uses_channel': True,  'use_ris': True,  'aircomp': False},
        'FedAvg (No RIS)': {'uses_channel': True,  'use_ris': False, 'aircomp': False},
        'AirComp + RIS':   {'uses_channel': True,  'use_ris': True,  'aircomp': True},
        'AirComp (No RIS)':{'uses_channel': True,  'use_ris': False, 'aircomp': True},
    }

    alpha_results = {}  # method -> alpha -> results
    fairness_results = {}  # method -> alpha -> jain's index

    short_rounds = min(exp_cfg.get('exp3_fl_rounds', 50), num_rounds)

    for method_name, method_cfg in methods.items():
        alpha_results[method_name] = {}
        fairness_results[method_name] = {}

        for alpha in alpha_sweep:
            print(f"  {method_name} | alpha={alpha}")
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Partition data
            if alpha >= 50:
                partition_map = iid_partition(train_dataset, num_clients)
            else:
                partition_map = dirichlet_noniid_partition(
                    train_dataset, num_clients, alpha)

            client_loaders = get_client_dataloaders(
                train_dataset, partition_map, batch_size,
                data_cfg.get('max_samples_per_client'))

            # Create clients
            clients = []
            for cid in range(num_clients):
                clients.append(FLClient(
                    client_id=cid,
                    dataloader=client_loaders[cid],
                    model_fn=model_fn,
                    config={
                        'lr': fl_cfg.get('learning_rate', 0.01),
                        'momentum': fl_cfg.get('momentum', 0.9),
                        'local_epochs': fl_cfg.get('local_epochs', 5),
                    }
                ))

            # Setup channel and the aggregation scheme used by this run.
            N_ris = ch_cfg.get('num_ris_elements', 64)
            run_cfg = copy.deepcopy(config)
            run_cfg.setdefault('aircomp', {})['enabled'] = method_cfg['aircomp']
            if method_cfg['uses_channel']:
                channel = WirelessChannel(run_cfg)
                ris_opt = RISOptimizer(num_elements=N_ris)
            else:
                channel = None
                ris_opt = None

            run_name = f"exp3_{method_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')}_alpha_{alpha}"
            logger = MetricsLogger(save_dir, run_name)
            server = FLServer(
                model_fn=model_fn,
                test_loader=test_loader,
                config=run_cfg,
                channel=channel,
                ris_optimizer=ris_opt if method_cfg.get('use_ris', False) else None,
                aircomp_aggregator=AirCompAggregator(run_cfg) if method_cfg['aircomp'] else None,
            )

            res = server.run(
                clients=clients,
                num_rounds=short_rounds,
                metrics_logger=logger,
                use_ris=method_cfg.get('use_ris', False),
                ris_mode='aircomp' if method_cfg['aircomp'] and method_cfg['use_ris'] else 'optimized',
                verbose=False,
            )
            logger.save_csv()
            logger.save_json()

            alpha_results[method_name][alpha] = res

            # Fairness is based on actual successful uploads, not data-size proxy.
            jain = res['successful_upload_fairness'][-1] if res['successful_upload_fairness'] else 0.0
            fairness_results[method_name][alpha] = jain

    # ---- Generate all plots ---- #
    print("\nGenerating plots...")
    _generate_plots(mse_vs_n, mse_vs_snr, alpha_results, fairness_results,
                    N_sweep, snr_sweep, alpha_sweep, methods,
                    save_dir, config)

    summary = {
        'aircomp_mse_vs_ris_elements': dict(zip(map(str, N_sweep), map(float, mse_vs_n))),
        'aircomp_mse_vs_snr_db': dict(zip(map(str, snr_sweep), map(float, mse_vs_snr))),
        'final_accuracy_percent': {
            method: {str(alpha): float(res['accuracy'][-1])
                     for alpha, res in values.items()}
            for method, values in alpha_results.items()
        },
        'successful_upload_fairness': {
            method: {str(alpha): float(value) for alpha, value in values.items()}
            for method, values in fairness_results.items()
        },
        'mean_snr_db': {
            method: {str(alpha): float(np.mean(res['mean_snr_db'])) if res.get('mean_snr_db') else 0.0
                     for alpha, res in values.items()}
            for method, values in alpha_results.items()
        },
        'mean_outage_probability': {
            method: {str(alpha): float(np.mean(res['outage_prob'])) if res.get('outage_prob') else 0.0
                     for alpha, res in values.items()}
            for method, values in alpha_results.items()
        },
        'mean_aggregation_mse': {
            method: {str(alpha): float(np.mean(res['aggregation_mse'])) if res.get('aggregation_mse') else 0.0
                     for alpha, res in values.items()}
            for method, values in alpha_results.items()
        },
        'mean_latency_ms': {
            method: {str(alpha): float(np.mean(res['per_round_latency_ms'])) if res.get('per_round_latency_ms') else 0.0
                     for alpha, res in values.items()}
            for method, values in alpha_results.items()
        },
        'mean_energy_mj': {
            method: {str(alpha): float(np.mean(res['mean_energy_mj'])) if res.get('mean_energy_mj') else 0.0
                     for alpha, res in values.items()}
            for method, values in alpha_results.items()
        },
    }
    with open(os.path.join(save_dir, 'exp3_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- Summary ---- #
    print(f"\n{'='*80}")
    print(f"{'Method':<20} | {'Acc(a=0.1)':>10} | {'Acc(IID)':>10} | {'Fairness(a=0.1)':>15}")
    print(f"{'-'*80}")
    for method_name in methods:
        acc_01 = alpha_results[method_name].get(0.1, {}).get('accuracy', [0])[-1]
        acc_iid = alpha_results[method_name].get(100.0, {}).get('accuracy', [0])[-1]
        fair = fairness_results[method_name].get(0.1, 0)
        print(f"{method_name:<20} | {acc_01:>9.2f}% | {acc_iid:>9.2f}% | {fair:>14.4f}")
    print(f"{'='*80}")


def _aircomp_mse_sweep(sweep_values, num_clients, config, sweep_type='N', num_trials=50):
    """Compute AirComp aggregation MSE across a parameter sweep.

    This simulates the AirComp process at the signal level without full FL.
    """
    mse_list = []
    aircomp = AirCompAggregator({})
    gradient_dim = 1000  # simulate with 1000-dimensional gradient vector

    for val in sweep_values:
        trial_mses = []

        for trial in range(num_trials):
            # Generate random "true" gradients
            true_gradients = [np.random.randn(gradient_dim) for _ in range(num_clients)]
            true_avg = np.mean(true_gradients, axis=0)

            # Setup channel with appropriate N
            ch_config = config.copy()
            if sweep_type == 'N':
                ch_config_ch = config.get('channel', {}).copy()
                ch_config_ch['num_ris_elements'] = val
                ch_config['channel'] = ch_config_ch
                channel = WirelessChannel(ch_config)
                ris_opt = RISOptimizer(num_elements=val, max_iters=20)
                tx_power_dbm = config.get('selection', {}).get('tx_power_dbm', 23.0)
            else:  # SNR sweep — adjust tx power to achieve target SNR
                channel = WirelessChannel(config)
                N_ris = config.get('channel', {}).get('num_ris_elements', 64)
                ris_opt = RISOptimizer(num_elements=N_ris, max_iters=20)
                # Map target SNR to tx power (approximate)
                tx_power_dbm = val + 10  # rough mapping

            # Generate channels
            ch_state = channel.generate_all_channels(num_clients)
            phi = ris_opt.optimize_for_aircomp(
                ch_state['h_br'], ch_state['h_ru'], ch_state['h_d'])

            # Compute effective channel gains for AirComp
            channel_gains = []
            for k in range(num_clients):
                h_eff = channel.compute_effective_channel(
                    ch_state['h_br'], ch_state['h_ru'][k], ch_state['h_d'][k], phi)
                # Scalar gain (sum across antennas)
                channel_gains.append(complex(np.sum(h_eff)))

            # Power allocation
            max_power = 10 ** ((tx_power_dbm - 30) / 10.0)
            # AirComp aggregation
            estimated_avg = aircomp.aggregate(
                true_gradients, channel_gains, channel.noise_power_w, max_power)

            if estimated_avg is not None:
                mse = aircomp.compute_aggregation_mse(true_avg, estimated_avg)
                trial_mses.append(mse)

        avg_mse = np.mean(trial_mses) if trial_mses else 1.0
        mse_list.append(avg_mse)
        print(f"  {sweep_type}={val}: MSE={avg_mse:.6f}")

    return mse_list


def _jain_fairness(values):
    """Jain's fairness index: J = (sum x_i)^2 / (n * sum x_i^2)."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0 or np.sum(arr**2) == 0:
        return 0
    return float((np.sum(arr))**2 / (n * np.sum(arr**2)))


def _generate_plots(mse_vs_n, mse_vs_snr, alpha_results, fairness_results,
                    N_sweep, snr_sweep, alpha_sweep, methods,
                    save_dir, config):
    """Generate all Experiment 3 plots."""
    plot_cfg = config.get('plot', {})
    dpi = plot_cfg.get('dpi', 300)
    fontsize = plot_cfg.get('font_size', 14)
    fmt = plot_cfg.get('format', 'pdf')

    plt.rcParams.update({'font.size': fontsize})
    sns.set_style("whitegrid")
    colors = sns.color_palette("Set2", max(len(methods), len(alpha_sweep)))

    # 1. MSE vs N (log scale)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(N_sweep, mse_vs_n, 'b-o', linewidth=2, markersize=8)
    ax.set_yscale('log')
    ax.set_xlabel('Number of RIS Elements (N)')
    ax.set_ylabel('Aggregation MSE (normalised)')
    ax.set_title('AirComp Error vs RIS Elements')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'mse_vs_n.{fmt}'), dpi=dpi)
    plt.close(fig)

    # 2. MSE vs SNR
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(snr_sweep, mse_vs_snr, 'g-s', linewidth=2, markersize=8)
    ax.set_yscale('log')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Aggregation MSE (normalised)')
    ax.set_title('AirComp Error vs SNR')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'mse_vs_snr.{fmt}'), dpi=dpi)
    plt.close(fig)

    # 3. Convergence for different alpha (using FedAvg+RIS)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, alpha in enumerate(alpha_sweep):
        label = 'IID' if alpha >= 50 else f'Dirichlet α={alpha}'
        method = 'FedAvg + RIS'
        if method in alpha_results and alpha in alpha_results[method]:
            res = alpha_results[method][alpha]
            ax.plot(res['round'], res['accuracy'], label=label,
                    color=colors[i], linewidth=2)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('FedAvg+RIS: Impact of Data Heterogeneity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'convergence_vs_alpha.{fmt}'), dpi=dpi)
    plt.close(fig)

    # 4. Bar chart: Final accuracy across methods and alpha values
    fig, ax = plt.subplots(figsize=(10, 6))
    method_names = list(methods.keys())
    bar_width = 0.2
    x = np.arange(len(alpha_sweep))

    for i, method in enumerate(method_names):
        accs = []
        for alpha in alpha_sweep:
            if alpha in alpha_results.get(method, {}):
                acc = alpha_results[method][alpha]['accuracy'][-1]
            else:
                acc = 0
            accs.append(acc)
        ax.bar(x + i * bar_width, accs, bar_width, label=method, color=colors[i])

    ax.set_xlabel('Data Heterogeneity')
    ax.set_ylabel('Final Test Accuracy (%)')
    ax.set_title('Performance Comparison Across Data Distributions')
    labels = ['IID' if a >= 50 else f'α={a}' for a in alpha_sweep]
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'acc_bar_comparison.{fmt}'), dpi=dpi)
    plt.close(fig)

    # 5. Fairness vs alpha
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, method in enumerate(method_names):
        jains = [fairness_results.get(method, {}).get(a, 0) for a in alpha_sweep]
        ax.plot(range(len(alpha_sweep)), jains, '-o', label=method,
                color=colors[i], linewidth=2, markersize=8)
    labels = ['IID' if a >= 50 else f'α={a}' for a in alpha_sweep]
    ax.set_xticks(range(len(alpha_sweep)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Data Heterogeneity (Dirichlet α)')
    ax.set_ylabel("Jain's Fairness Index")
    ax.set_title('Fairness Across Data Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'fairness_vs_alpha.{fmt}'), dpi=dpi)
    plt.close(fig)

    print(f"  Plots saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Experiment 3: AirComp + non-IID study")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    run_experiment(config)


if __name__ == "__main__":
    main()
