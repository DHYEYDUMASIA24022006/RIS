"""Experiment 1: Baseline FL with/without RIS.

Runs FL under 4 conditions:
  (a) Ideal channel (no fading, no RIS) — upper bound
  (b) Fading channel, no RIS — lower bound
  (c) Fading channel + random RIS phases
  (d) Fading channel + optimised RIS phases

Measures: test accuracy vs round, rounds to target accuracy, outage probability, mean SNR.
Generates publication-quality convergence comparison plots.

Usage:
    python experiments/exp1_baseline.py
    python experiments/exp1_baseline.py --config config/config.yaml
"""

import argparse
import copy
import os
import sys
import yaml
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel.channel_model import WirelessChannel
from channel.ris_optimizer import RISOptimizer
from fl.models import get_model
from fl.data_partition import (load_dataset, iid_partition,
                               dirichlet_noniid_partition, get_client_dataloaders)
from fl.client import FLClient
from fl.server import FLServer
from metrics.metrics_logger import MetricsLogger
from utils.plotting import save_figure


def run_experiment(config: dict):
    """Run Experiment 1: Baseline FL with/without RIS."""

    # ---- Setup ---- #
    seed = config.get('system', {}).get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Isolate Exp1: every client attempts the uplink; fading/outage drop packets.
    cfg = copy.deepcopy(config)
    cfg.setdefault('fl', {})
    cfg['fl']['fraction_fit'] = 1.0
    cfg['fl']['use_rate_outage'] = False
    cfg['fl']['enforce_energy_latency'] = False
    cfg.setdefault('selection', {})
    cfg['selection']['strategy'] = 'all'

    fl_cfg = cfg.get('fl', {})
    ch_cfg = cfg.get('channel', {})
    data_cfg = cfg.get('data', {})
    exp_cfg = cfg.get('experiments', {})

    num_clients = fl_cfg.get('num_clients', 20)
    num_rounds = exp_cfg.get('exp1_rounds', fl_cfg.get('num_rounds', 100))
    batch_size = fl_cfg.get('batch_size', 32)
    dataset_name = data_cfg.get('dataset', 'mnist')
    N_ris = ch_cfg.get('num_ris_elements', 64)

    save_dir = os.path.join(cfg.get('system', {}).get('results_dir', 'results'), 'exp1')
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Experiment 1: Baseline FL with/without RIS")
    print(f"Dataset: {dataset_name} | Clients: {num_clients} | Rounds: {num_rounds}")
    print(f"RIS elements: {N_ris} | Device: {device}")
    print(f"Direct blockage: p={ch_cfg.get('direct_blockage_prob', 0)}, "
          f"+{ch_cfg.get('direct_blockage_attenuation_db', 0)} dB")
    print(f"Outage SNR threshold: {fl_cfg.get('snr_outage_threshold_db', 5)} dB | "
          f"uplink noise scale: {fl_cfg.get('update_noise_scale', 0)}")
    print(f"{'='*60}\n")

    # ---- Data ---- #
    train_dataset, test_dataset = load_dataset(dataset_name)
    partition_mode = data_cfg.get('partition', 'iid').lower()
    if partition_mode == 'noniid':
        alpha_values = data_cfg.get('dirichlet_alpha', [0.5])
        alpha = float(data_cfg.get('exp1_dirichlet_alpha', alpha_values[0]))
        partition_map = dirichlet_noniid_partition(train_dataset, num_clients, alpha)
        print(f"Data partition: non-IID Dirichlet (alpha={alpha})")
    elif partition_mode == 'iid':
        partition_map = iid_partition(train_dataset, num_clients)
        print("Data partition: IID")
    else:
        raise ValueError("data.partition must be 'iid' or 'noniid'")
    client_loaders = get_client_dataloaders(
        train_dataset, partition_map, batch_size,
        data_cfg.get('max_samples_per_client'))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)

    # ---- Model factory ---- #
    def model_fn():
        return get_model(dataset_name, torch.device('cpu'))

    # ---- Channel & RIS ---- #
    channel = WirelessChannel(cfg)
    ris_opt = RISOptimizer(num_elements=N_ris,
                           max_iters=cfg.get('ris', {}).get('max_iterations', 50),
                           tol=cfg.get('ris', {}).get('convergence_threshold', 1e-6))

    # ---- Conditions ---- #
    conditions = {
        'Ideal (No Fading)':     {'channel': None,    'use_ris': False, 'ris_mode': 'none'},
        'Fading (No RIS)':       {'channel': channel, 'use_ris': False, 'ris_mode': 'none'},
        'Fading + Random RIS':   {'channel': channel, 'use_ris': True,  'ris_mode': 'random'},
        'Fading + Optimised RIS':{'channel': channel, 'use_ris': True,  'ris_mode': 'optimized'},
    }

    all_results = {}

    for cond_name, cond_params in conditions.items():
        print(f"\n--- Condition: {cond_name} ---")
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create fresh clients
        clients = []
        for cid in range(num_clients):
            client = FLClient(
                client_id=cid,
                dataloader=client_loaders[cid],
                model_fn=model_fn,
                config={
                    'lr': fl_cfg.get('learning_rate', 0.01),
                    'momentum': fl_cfg.get('momentum', 0.9),
                    'local_epochs': fl_cfg.get('local_epochs', 1),
                }
            )
            clients.append(client)

        # Create server
        logger = MetricsLogger(save_dir, f"exp1_{cond_name.replace(' ', '_')}")
        server = FLServer(
            model_fn=model_fn,
            test_loader=test_loader,
            config=cfg,
            channel=cond_params['channel'],
            ris_optimizer=ris_opt if cond_params['use_ris'] else None,
        )

        # Run FL
        results = server.run(
            clients=clients,
            num_rounds=num_rounds,
            metrics_logger=logger,
            use_ris=cond_params['use_ris'],
            ris_mode=cond_params['ris_mode'],
            verbose=True,
        )

        logger.save_csv()
        logger.save_json()
        all_results[cond_name] = results

    # ---- Generate plots ---- #
    print("\nGenerating plots...")
    _generate_plots(all_results, save_dir, cfg)

    # ---- Summary table ---- #
    target = fl_cfg.get('target_accuracy', 0.90) * 100
    print(f"\n{'='*110}")
    print(f"{'Condition':<25} | {'Final Acc':>9} | {'Rnds@'+str(int(target))+'%':>8} | "
          f"{'SNR dB':>7} | {'Outage':>7} | {'Succ/rnd':>8} | {'Agg MSE':>8}")
    print(f"{'-'*110}")
    for name, res in all_results.items():
        final_acc = res['accuracy'][-1] if res['accuracy'] else 0
        rounds_target = _get_rounds_to_target(res['accuracy'], target)
        mean_snr = np.mean(res['mean_snr_db']) if res['mean_snr_db'] else float('inf')
        outage = np.mean(res['outage_prob']) if res['outage_prob'] else 0
        succ = np.mean(res.get('num_successful', res.get('num_participating', [0])))
        mse = np.mean(res.get('aggregation_mse', [0]))
        print(f"{name:<25} | {final_acc:>8.2f}% | {str(rounds_target):>8} | "
              f"{mean_snr:>7.1f} | {outage:>7.3f} | {succ:>8.1f} | {mse:>8.2e}")
    print(f"{'='*110}")


def _get_rounds_to_target(accuracy_list, target):
    """Return first round achieving target accuracy, or 'N/A'."""
    for i, acc in enumerate(accuracy_list):
        if acc >= target:
            return str(i + 1)
    return 'N/A'


def _generate_plots(results: dict, save_dir: str, config: dict):
    """Generate publication-quality comparison plots."""
    plot_cfg = config.get('plot', {})
    dpi = plot_cfg.get('dpi', 300)
    fontsize = plot_cfg.get('font_size', 14)
    fmt = plot_cfg.get('format', 'pdf')

    plt.rcParams.update({
        'font.size': fontsize,
        'axes.labelsize': fontsize,
        'axes.titlesize': fontsize + 2,
        'legend.fontsize': fontsize - 2,
        'xtick.labelsize': fontsize - 2,
        'ytick.labelsize': fontsize - 2,
    })
    sns.set_style("whitegrid")
    colors = sns.color_palette("Set2", len(results))
    markers = ['o', 's', '^', 'D', 'v', '<']

    conditions = list(results.keys())

    # 1. Convergence curves
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cond in enumerate(conditions):
        acc = results[cond]['accuracy']
        rounds = results[cond]['round']
        ax.plot(rounds, acc, label=cond, color=colors[i],
                linewidth=2, marker=markers[i % len(markers)],
                markevery=max(1, len(rounds)//10), markersize=6)

    target = config.get('fl', {}).get('target_accuracy', 0.90) * 100
    ax.axhline(y=target, color='red', linestyle='--', alpha=0.6,
               label=f'{int(target)}% Target')
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('FL Convergence: Impact of RIS')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'convergence.{fmt}'), dpi=dpi)
    plt.close(fig)

    # 2. Rounds to target accuracy (bar chart)
    fig, ax = plt.subplots(figsize=(8, 6))
    rounds_to_target = []
    for cond in conditions:
        r = _get_rounds_to_target(results[cond]['accuracy'], target)
        rounds_to_target.append(int(r) if r != 'N/A' else results[cond]['round'][-1])
    bars = ax.bar(conditions, rounds_to_target, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel(f'Rounds to {int(target)}% Accuracy')
    ax.set_title('Communication Efficiency')
    plt.xticks(rotation=15, ha='right')
    for bar, val in zip(bars, rounds_to_target):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=fontsize - 2)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'rounds_to_target.{fmt}'), dpi=dpi)
    plt.close(fig)

    # 3. SNR CDF
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cond in enumerate(conditions):
        snrs = results[cond].get('mean_snr_db', [])
        if snrs and not all(s > 100 for s in snrs):  # skip ideal
            sorted_snrs = np.sort(snrs)
            cdf = np.arange(1, len(sorted_snrs) + 1) / len(sorted_snrs)
            ax.plot(sorted_snrs, cdf, label=cond, color=colors[i], linewidth=2)
    ax.set_xlabel('Mean SNR (dB)')
    ax.set_ylabel('CDF')
    ax.set_title('SNR Distribution Across Rounds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'snr_cdf.{fmt}'), dpi=dpi)
    plt.close(fig)

    # 4. Outage probability over rounds
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cond in enumerate(conditions):
        outage = results[cond].get('outage_prob', [])
        if outage:
            ax.plot(results[cond]['round'], outage, label=cond,
                    color=colors[i], linewidth=2)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Outage Probability')
    ax.set_title('Outage Probability Over Training')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'outage_prob.{fmt}'), dpi=dpi)
    plt.close(fig)

    # 5. Successful uploads per round
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cond in enumerate(conditions):
        succ = results[cond].get('num_successful', results[cond].get('num_participating', []))
        if succ:
            ax.plot(results[cond]['round'], succ, label=cond,
                    color=colors[i], linewidth=2, marker=markers[i % len(markers)],
                    markevery=max(1, len(succ)//10))
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Successful Uplink Clients')
    ax.set_title('Dropped Updates: Channel Outage')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'successful_uploads.{fmt}'), dpi=dpi)
    plt.close(fig)

    # 6. Aggregation MSE (ideal channel is ~0)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cond in enumerate(conditions):
        mse = results[cond].get('aggregation_mse', [])
        if mse and not all(m == 0 for m in mse):
            ax.semilogy(results[cond]['round'], np.maximum(mse, 1e-16),
                        label=cond, color=colors[i], linewidth=2)
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Uplink Parameter MSE')
    ax.set_title('Aggregation Distortion vs Channel Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'aggregation_mse.{fmt}'), dpi=dpi)
    plt.close(fig)

    print(f"  Plots saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Baseline FL with/without RIS")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Config not found at {args.config}, using defaults.")
        config = {}

    run_experiment(config)


if __name__ == "__main__":
    main()
