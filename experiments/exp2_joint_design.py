"""Experiment 2: Joint Device Selection + RIS Design.

Compares the proposed joint design, three priority-term ablations, and three
channel/random baselines under the same partition, seed, participant cap, and
resource budgets.

Sweeps:
  - Latency budget T_max
  - Energy budget E_max

Measures: test accuracy, num participating clients, per-round latency, energy.

Usage:
    python experiments/exp2_joint_design.py
    python experiments/exp2_joint_design.py --config config/config.yaml
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
from fl.data_partition import (load_dataset_with_validation, iid_partition,
                               dirichlet_noniid_partition, get_client_dataloaders)
from fl.client import FLClient
from fl.server import FLServer
from optimisation.joint_design import JointDeviceRISOptimizer
from optimisation.baselines import RandomSelector, ChannelOnlySelector
from metrics.metrics_logger import MetricsLogger
from utils.plotting import save_figure


def run_experiment(config: dict):
    """Run Experiment 2: Joint Device Selection + RIS Design."""

    seed = config.get('system', {}).get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fl_cfg = config.get('fl', {})
    ch_cfg = config.get('channel', {})
    data_cfg = config.get('data', {})
    exp_cfg = config.get('experiments', {})
    sel_cfg = config.get('selection', {})

    num_clients = fl_cfg.get('num_clients', 20)
    num_rounds = fl_cfg.get('num_rounds', 100)
    batch_size = fl_cfg.get('batch_size', 32)
    dataset_name = data_cfg.get('dataset', 'mnist')
    N_ris = ch_cfg.get('num_ris_elements', 64)

    latency_sweep = exp_cfg.get('exp2_latency_sweep', [50, 75, 100, 150, 200])
    energy_sweep = exp_cfg.get('exp2_energy_sweep', [20, 30, 50, 75, 100])

    save_dir = os.path.join(config.get('system', {}).get('results_dir', 'results'), 'exp2')
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Experiment 2: Joint Device Selection + RIS Design")
    print(f"Dataset: {dataset_name} | Clients: {num_clients} | Rounds: {num_rounds}")
    print(f"{'='*60}\n")

    # ---- Data ---- #
    validation_fraction = float(data_cfg.get('validation_fraction', 0.05))
    (train_dataset, validation_dataset, test_dataset,
     train_indices, validation_indices) = load_dataset_with_validation(
        dataset_name, validation_fraction, seed)
    partition = str(data_cfg.get('partition', 'noniid')).lower()
    alpha = float(data_cfg.get('exp2_dirichlet_alpha',
                               data_cfg.get('exp1_dirichlet_alpha', 0.5)))
    if partition == 'iid':
        partition_map = iid_partition(train_dataset, num_clients, train_indices)
    elif partition == 'noniid':
        partition_map = dirichlet_noniid_partition(
            train_dataset, num_clients, alpha,
            int(data_cfg.get('num_classes', 10)), train_indices)
    else:
        raise ValueError("data.partition must be 'iid' or 'noniid'")
    client_loaders = get_client_dataloaders(
        train_dataset, partition_map, batch_size,
        data_cfg.get('max_samples_per_client'))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False)
    validation_subset = torch.utils.data.Subset(validation_dataset, validation_indices)
    validation_limit = int(sel_cfg.get('utility_validation_samples', 512))
    if validation_limit > 0 and len(validation_subset) > validation_limit:
        validation_subset = torch.utils.data.Subset(
            validation_dataset, validation_indices[:validation_limit])
    validation_loader = torch.utils.data.DataLoader(
        validation_subset, batch_size=128, shuffle=False)

    def model_fn():
        return get_model(dataset_name, torch.device('cpu'))

    # ---- Channel & RIS ---- #
    channel = WirelessChannel(config)
    ris_opt = RISOptimizer(num_elements=N_ris,
                           max_iters=config.get('ris', {}).get('max_iterations', 50))

    strategies = {
        'Joint+RIS (Proposed)': 'joint',
        'Joint+RIS (No utility)': 'joint_no_utility',
        'Joint+RIS (No upload debt)': 'joint_no_debt',
        'Joint+RIS (No starvation age)': 'joint_no_age',
        'Channel-Only (No RIS)': 'channel_only',
        'Random (No RIS)': 'random',
        'Channel + Random RIS': 'channel_random_ris',
    }

    # ============================================================= #
    # Part 1: Fixed budget, compare convergence                       #
    # ============================================================= #
    fixed_latency = sel_cfg.get('latency_budget_ms', 100.0)
    fixed_energy = sel_cfg.get('energy_budget_mj', 50.0)

    print(f"Part 1: Fixed budget (T_max={fixed_latency}ms, E_max={fixed_energy}mJ)")
    convergence_results = {}

    for strat_name, strat_type in strategies.items():
        print(f"\n  Strategy: {strat_name}")
        cache_path = os.path.join(save_dir, f"exp2_{strat_type}.json")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                if len(cached_data.get('round', [])) >= num_rounds:
                    print(f"    [Cache] Loaded completed {num_rounds} rounds from {cache_path}")
                    convergence_results[strat_name] = cached_data
                    continue
            except Exception:
                pass

        np.random.seed(seed)
        torch.manual_seed(seed)

        clients = _create_clients(num_clients, client_loaders, model_fn, fl_cfg, config)

        run_cfg = _exp2_config(config, fixed_latency, fixed_energy)
        run_cfg['selection']['strategy'] = strat_type
        selector = _get_selector(strat_type, run_cfg, channel, ris_opt)
        use_ris, ris_mode = _ris_flags(strat_type)

        server = FLServer(
            model_fn=model_fn,
            test_loader=test_loader,
            config=run_cfg,
            channel=channel,
            ris_optimizer=ris_opt if use_ris else None,
            device_selector=selector,
            validation_loader=validation_loader,
        )
        logger = MetricsLogger(save_dir, f"exp2_{strat_type}")

        results = server.run(
            clients=clients,
            num_rounds=num_rounds,
            metrics_logger=logger,
            use_ris=use_ris,
            ris_mode=ris_mode,
            verbose=True,
        )
        logger.save_csv()
        logger.save_json()
        convergence_results[strat_name] = results

    skip_sweeps = bool(exp_cfg.get('exp2_skip_sweeps', False))
    latency_results = {s: [] for s in strategies}
    energy_results = {s: [] for s in strategies}

    # ============================================================= #
    # Part 2: Sweep latency budget (fixed energy)                     #
    # ============================================================= #
    lat_sweep_cache = os.path.join(save_dir, 'exp2_latency_sweep.json')
    if not skip_sweeps:
        if os.path.exists(lat_sweep_cache):
            try:
                with open(lat_sweep_cache, 'r') as f:
                    cached_lat = json.load(f)
                if all(len(cached_lat.get(s, [])) == len(latency_sweep) for s in strategies):
                    print(f"\nPart 2: Latency sweep loaded from {lat_sweep_cache}")
                    latency_results = cached_lat
            except Exception:
                pass

        if not any(latency_results[s] for s in strategies):
            print(f"\nPart 2: Latency sweep (fixed E_max={fixed_energy}mJ)")
            sweep_rounds = min(int(exp_cfg.get('exp2_sweep_rounds', 30)), num_rounds)

            for t_max in latency_sweep:
                print(f"  T_max = {t_max} ms")
                for strat_name, strat_type in strategies.items():
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                    clients = _create_clients(num_clients, client_loaders, model_fn, fl_cfg, config)
                    run_cfg = _exp2_config(config, t_max, fixed_energy)
                    run_cfg['selection']['strategy'] = strat_type
                    selector = _get_selector(strat_type, run_cfg, channel, ris_opt)
                    use_ris, ris_mode = _ris_flags(strat_type)

                    server = FLServer(
                        model_fn=model_fn, test_loader=test_loader, config=run_cfg,
                        channel=channel, ris_optimizer=ris_opt if use_ris else None,
                        device_selector=selector,
                        validation_loader=validation_loader,
                    )
                    res = server.run(clients, sweep_rounds, use_ris=use_ris,
                                     ris_mode=ris_mode, verbose=False)
                    latency_results[strat_name].append(res['accuracy'][-1] if res['accuracy'] else 0)

            with open(lat_sweep_cache, 'w') as f:
                json.dump(latency_results, f, indent=2)

        # ============================================================= #
        # Part 3: Sweep energy budget (fixed latency)                     #
        # ============================================================= #
        eng_sweep_cache = os.path.join(save_dir, 'exp2_energy_sweep.json')
        if os.path.exists(eng_sweep_cache):
            try:
                with open(eng_sweep_cache, 'r') as f:
                    cached_eng = json.load(f)
                if all(len(cached_eng.get(s, [])) == len(energy_sweep) for s in strategies):
                    print(f"\nPart 3: Energy sweep loaded from {eng_sweep_cache}")
                    energy_results = cached_eng
            except Exception:
                pass

        if not any(energy_results[s] for s in strategies):
            print(f"\nPart 3: Energy sweep (fixed T_max={fixed_latency}ms)")
            sweep_rounds = min(int(exp_cfg.get('exp2_sweep_rounds', 30)), num_rounds)

            for e_max in energy_sweep:
                print(f"  E_max = {e_max} mJ")
                for strat_name, strat_type in strategies.items():
                    np.random.seed(seed)
                    torch.manual_seed(seed)

                    clients = _create_clients(num_clients, client_loaders, model_fn, fl_cfg, config)
                    run_cfg = _exp2_config(config, fixed_latency, e_max)
                    run_cfg['selection']['strategy'] = strat_type
                    selector = _get_selector(strat_type, run_cfg, channel, ris_opt)
                    use_ris, ris_mode = _ris_flags(strat_type)

                    server = FLServer(
                        model_fn=model_fn, test_loader=test_loader, config=run_cfg,
                        channel=channel, ris_optimizer=ris_opt if use_ris else None,
                        device_selector=selector,
                        validation_loader=validation_loader,
                    )
                    res = server.run(clients, sweep_rounds, use_ris=use_ris,
                                     ris_mode=ris_mode, verbose=False)
                    num_participating = np.mean(res['num_participating']) if res['num_participating'] else 0
                    energy_results[strat_name].append(num_participating)

            with open(eng_sweep_cache, 'w') as f:
                json.dump(energy_results, f, indent=2)
    else:
        print('\nSkipping Exp2 latency/energy sweeps (exp2_skip_sweeps=true).')

    # ---- Generate plots ---- #
    print("\nGenerating plots...")
    _generate_plots(convergence_results, latency_results, energy_results,
                    latency_sweep, energy_sweep, strategies, save_dir, config)

    summary = {
        'fixed_budget': {'latency_ms': fixed_latency, 'energy_mj': fixed_energy},
        'data': {'dataset': dataset_name, 'partition': partition,
                 'dirichlet_alpha': alpha if partition == 'noniid' else None,
                 'validation_fraction': validation_fraction},
        'convergence': {
            name: {
                'final_accuracy_percent': float(res['accuracy'][-1]),
                'best_accuracy_percent': float(max(res['accuracy'])) if res['accuracy'] else 0.0,
                'mean_participating_clients': float(np.mean(res['num_participating'])),
                'mean_successful_uploads': float(np.mean(res['num_successful'])),
                'mean_snr_db': float(np.mean(res['mean_snr_db'])) if res.get('mean_snr_db') else 0.0,
                'mean_outage_probability': float(np.mean(res['outage_prob'])) if res.get('outage_prob') else 0.0,
                'mean_aggregation_mse': float(np.mean(res['aggregation_mse'])) if res.get('aggregation_mse') else 0.0,
                'mean_latency_ms': float(np.mean(res['per_round_latency_ms'])),
                'mean_energy_mj': float(np.mean(res['mean_energy_mj'])),
                'successful_upload_fairness': float(res['successful_upload_fairness'][-1]),
                'idle_round_fraction': float(np.mean(res['idle_round'])) if res['idle_round'] else 0.0,
                'total_resource_constraint_violations': int(sum(res['resource_constraint_violations'])),
            }
            for name, res in convergence_results.items()
        },
        'accuracy_vs_latency_budget_ms': {
            name: dict(zip(map(str, latency_sweep), map(float, values)))
            for name, values in latency_results.items()
        },
        'participants_vs_energy_budget_mj': {
            name: dict(zip(map(str, energy_sweep), map(float, values)))
            for name, values in energy_results.items()
        },
    }
    with open(os.path.join(save_dir, 'exp2_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(save_dir, 'exp2_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # ---- Summary ---- #
    print(f"\n{'='*80}")
    print(f"{'Strategy':<30} | {'Final Acc':>10} | {'Avg Clients':>12} | {'Fairness':>10}")
    print(f"{'-'*80}")
    for name, res in convergence_results.items():
        acc = res['accuracy'][-1] if res['accuracy'] else 0
        clients_avg = np.mean(res['num_participating']) if res['num_participating'] else 0
        fairness = res['successful_upload_fairness'][-1] if res['successful_upload_fairness'] else 0
        print(f"{name:<30} | {acc:>9.2f}% | {clients_avg:>11.1f} | {fairness:>9.3f}")
    print(f"{'='*80}")


def _create_clients(num_clients, client_loaders, model_fn, fl_cfg, config):
    """Helper to create fresh FL clients."""
    clients = []
    for cid in range(num_clients):
        clients.append(FLClient(
            client_id=cid,
            dataloader=client_loaders[cid],
            model_fn=model_fn,
            config={
                'lr': float(fl_cfg.get('learning_rate', 0.01)),
                'momentum': float(fl_cfg.get('momentum', 0.9)),
                'local_epochs': int(fl_cfg.get('local_epochs', 5)),
                'lr_decay': float(fl_cfg.get('lr_decay', 1.0)),
                'weight_decay': float(fl_cfg.get('weight_decay', 0.0)),
                'device': config.get('system', {}).get('device', 'auto'),
            }
        ))
    return clients


def _exp2_config(config, latency_ms, energy_mj):
    """Copy config and enable latency/energy filtering for Methodology 2."""
    cfg = copy.deepcopy(config)
    cfg.setdefault('fl', {})
    cfg['fl']['use_rate_outage'] = True
    cfg['fl']['enforce_energy_latency'] = True
    cfg.setdefault('selection', {})
    cfg['selection']['latency_budget_ms'] = latency_ms
    cfg['selection']['energy_budget_mj'] = energy_mj
    return cfg


def _ris_flags(strat_type):
    use_ris = strat_type.startswith('joint') or strat_type == 'channel_random_ris'
    if strat_type.startswith('joint'):
        return use_ris, 'optimized'
    if strat_type == 'channel_random_ris':
        return use_ris, 'random'
    return use_ris, 'none'


def _get_selector(strat_type, config, channel, ris_opt):
    """Get appropriate device selector."""
    fl_cfg = config.get('fl', {})
    sel_cfg = config.get('selection', {})
    k = sel_cfg.get('max_participants')
    if k is None:
        k = max(1, int(fl_cfg.get('fraction_fit', 0.5) * fl_cfg.get('num_clients', 20)))
    k = int(k)
    if strat_type.startswith('joint'):
        selector_config = copy.deepcopy(config)
        selection = selector_config.setdefault('selection', {})
        if strat_type == 'joint_no_utility':
            selection['learning_utility_weight'] = 0.0
        elif strat_type == 'joint_no_debt':
            selection['participation_debt_weight'] = 0.0
        elif strat_type == 'joint_no_age':
            selection['starvation_age_weight'] = 0.0
        return JointDeviceRISOptimizer(channel, ris_opt, selector_config)
    if strat_type == 'random':
        return RandomSelector(k)
    return ChannelOnlySelector(k)


def _generate_plots(convergence_results, latency_results, energy_results,
                    latency_sweep, energy_sweep, strategies, save_dir, config):
    """Generate all Experiment 2 plots."""
    plot_cfg = config.get('plot', {})
    dpi = plot_cfg.get('dpi', 300)
    fontsize = plot_cfg.get('font_size', 14)
    fmt = plot_cfg.get('format', 'pdf')

    plt.rcParams.update({'font.size': fontsize})
    sns.set_style("whitegrid")
    colors = sns.color_palette("Set2", len(strategies))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

    strat_names = list(strategies.keys())

    # 1. Convergence comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(strat_names):
        res = convergence_results[name]
        ax.plot(res['round'], res['accuracy'], label=name, color=colors[i],
                linewidth=2, marker=markers[i], markevery=max(1, len(res['round'])//10))
    ax.set_xlabel('Communication Round')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title('Convergence: Selection Strategies')
    ax.legend(loc='lower right', fontsize=fontsize-3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'convergence.{fmt}'), dpi=dpi)
    save_figure(fig, os.path.join(save_dir, 'convergence.png'), dpi=dpi)
    plt.close(fig)

    # 2. Accuracy vs Latency Budget
    if any(latency_results[name] for name in strat_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, name in enumerate(strat_names):
            ax.plot(latency_sweep, latency_results[name], label=name, color=colors[i],
                    linewidth=2, marker=markers[i], markersize=8)
        ax.set_xlabel('Latency Budget $T_{max}$ (ms)')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Accuracy vs Latency Budget')
        ax.set_ylim(bottom=0, top=100)
        ax.legend(fontsize=fontsize-3)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_figure(fig, os.path.join(save_dir, f'acc_vs_latency.{fmt}'), dpi=dpi)
        save_figure(fig, os.path.join(save_dir, 'acc_vs_latency.png'), dpi=dpi)
        plt.close(fig)

    # 3. Clients selected vs Energy Budget
    if any(energy_results[name] for name in strat_names):
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, name in enumerate(strat_names):
            ax.plot(energy_sweep, energy_results[name], label=name, color=colors[i],
                    linewidth=2, marker=markers[i], markersize=8)
        ax.set_xlabel('Energy Budget $E_{max}$ (mJ)')
        ax.set_ylabel('Avg. Participating Clients')
        ax.set_title('Client Participation vs Energy Budget')
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=fontsize-3)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_figure(fig, os.path.join(save_dir, f'clients_vs_energy.{fmt}'), dpi=dpi)
        save_figure(fig, os.path.join(save_dir, 'clients_vs_energy.png'), dpi=dpi)
        plt.close(fig)

    # 4. Latency CDF
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(strat_names):
        lat = convergence_results[name].get('per_round_latency_ms', [])
        valid_lat = [float(l) for l in lat if l is not None and np.isfinite(l) and l > 0]
        if valid_lat:
            sorted_lat = np.sort(valid_lat)
            cdf = np.arange(1, len(sorted_lat)+1) / len(sorted_lat)
            ax.plot(sorted_lat, cdf, label=name, color=colors[i], linewidth=2)
    ax.set_xlabel('Per-Round Latency (ms)')
    ax.set_ylabel('CDF')
    ax.set_title('Latency Distribution')
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=fontsize-3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_dir, f'latency_cdf.{fmt}'), dpi=dpi)
    save_figure(fig, os.path.join(save_dir, 'latency_cdf.png'), dpi=dpi)
    plt.close(fig)

    print(f"  Plots saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: Joint Device Selection + RIS")
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
