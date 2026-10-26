"""Unit tests for RIS-Assisted Wireless Federated Learning.

Tests components across the pipeline:
1. Wireless Channel (Path loss, Rician/Rayleigh channel generation, SNR, achievable rate)
2. RIS Optimizer (Element-wise Alternating Optimization for Sum-SNR, Min-SNR, AirComp)
3. Feasibility & Resource Estimation (Latency, Energy, Model size bits, Constraint checks)
4. Joint Device & RIS Selector (Ablations, Priority ranking, Greedy selection, Fallbacks)
5. Metric Logger & Fairness (Jain's index, metric recording)
6. Model Architecture & CIFAR-10 Compatibility
"""

import unittest
import numpy as np
import copy
import yaml
import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel.channel_model import WirelessChannel
from channel.ris_optimizer import RISOptimizer
from optimisation.joint_design import JointDeviceRISOptimizer
from optimisation.baselines import RandomSelector, ChannelOnlySelector
from metrics.metrics_logger import MetricsLogger


def get_test_config():
    """Load default config for testing."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {
            'system': {'seed': 42, 'device': 'cpu'},
            'channel': {
                'model': 'rician', 'num_antennas_bs': 4, 'num_ris_elements': 64,
                'rician_k_factor': 10.0, 'carrier_freq_ghz': 3.5, 'bandwidth_mhz': 10.0,
                'noise_power_dbm': -174, 'path_loss_exponent_direct': 3.5,
                'path_loss_exponent_br': 2.2, 'path_loss_exponent_ru': 2.5,
                'd_bs_ris': 20.0, 'd_ris_user_min': 5.0, 'd_ris_user_max': 50.0,
                'd_bs_user_min': 30.0, 'd_bs_user_max': 150.0,
                'direct_blockage_prob': 0.80, 'direct_blockage_attenuation_db': 35.0
            },
            'fl': {
                'num_clients': 20, 'num_rounds': 50, 'local_epochs': 5,
                'batch_size': 32, 'fraction_fit': 0.5, 'snr_outage_threshold_db': 8.0,
                'use_rate_outage': True, 'enforce_energy_latency': True
            },
            'selection': {
                'strategy': 'joint', 'latency_budget_ms': 100.0, 'energy_budget_mj': 50.0,
                'tx_power_dbm': 23.0, 'computation_capacity_ghz': 1.0, 'cycles_per_sample': 20.0,
                'computation_energy_per_sample_mj': 0.001, 'max_participants': 10,
                'learning_utility_weight': 0.65, 'participation_debt_weight': 0.20,
                'starvation_age_weight': 0.15, 'starvation_age_cap_rounds': 100,
                'utility_ema_momentum': 0.8, 'max_alternating_iters': 5
            }
        }
    return cfg


class TestWirelessChannel(unittest.TestCase):
    """Test wireless channel model equations and dimensions."""

    def setUp(self):
        self.config = get_test_config()
        self.channel = WirelessChannel(self.config)
        self.num_clients = 20

    def test_channel_dimensions(self):
        state = self.channel.generate_all_channels(self.num_clients)
        M = self.config['channel']['num_antennas_bs']
        N = self.config['channel']['num_ris_elements']

        self.assertEqual(state['h_br'].shape, (M, N))
        self.assertEqual(len(state['h_ru']), self.num_clients)
        self.assertEqual(state['h_ru'][0].shape, (N, 1))
        self.assertEqual(len(state['h_d']), self.num_clients)
        self.assertEqual(state['h_d'][0].shape, (M, 1))

    def test_effective_channel_and_snr(self):
        state = self.channel.generate_all_channels(self.num_clients)
        phi = np.exp(1j * np.random.uniform(0, 2 * np.pi, self.config['channel']['num_ris_elements']))
        h_eff = self.channel.compute_effective_channel(state['h_br'], state['h_ru'][0], state['h_d'][0], phi)
        self.assertEqual(h_eff.shape, (self.config['channel']['num_antennas_bs'], 1))

        snr = self.channel.compute_snr(h_eff, 23.0)
        self.assertGreater(snr, 0)
        self.assertTrue(np.isfinite(snr))

        rate = self.channel.compute_achievable_rate(snr)
        self.assertGreater(rate, 0)
        self.assertTrue(np.isfinite(rate))

    def test_ris_gain_over_blocked_direct(self):
        """Optimized RIS phase should yield higher received power than blocked direct link."""
        state = self.channel.generate_all_channels(1)
        h_br = state['h_br']
        h_ru = state['h_ru'][0]
        h_d = state['h_d'][0]

        direct_gain = np.linalg.norm(h_d) ** 2

        opt = RISOptimizer(self.config['channel']['num_ris_elements'], max_iters=20)
        phi_opt = opt.optimize_sum_snr(h_br, [h_ru], [h_d])
        h_eff_opt = self.channel.compute_effective_channel(h_br, h_ru, h_d, phi_opt)
        ris_gain = np.linalg.norm(h_eff_opt) ** 2

        self.assertGreater(ris_gain, direct_gain)


class TestRISOptimizer(unittest.TestCase):
    """Test RIS phase optimization algorithms."""

    def setUp(self):
        self.config = get_test_config()
        self.N = self.config['channel']['num_ris_elements']
        self.opt = RISOptimizer(self.N, max_iters=15)
        self.channel = WirelessChannel(self.config)

    def test_phase_unit_modulus(self):
        phi_rand = self.opt.random_phases()
        self.assertEqual(len(phi_rand), self.N)
        np.testing.assert_allclose(np.abs(phi_rand), 1.0, atol=1e-6)

        state = self.channel.generate_all_channels(4)
        phi_opt = self.opt.optimize_sum_snr(state['h_br'], state['h_ru'], state['h_d'])
        self.assertEqual(len(phi_opt), self.N)
        np.testing.assert_allclose(np.abs(phi_opt), 1.0, atol=1e-6)

    def test_weighted_sum_snr_optimization(self):
        state = self.channel.generate_all_channels(3)
        weights = np.array([0.8, 0.1, 0.1])
        phi_opt = self.opt.optimize_sum_snr(state['h_br'], state['h_ru'], state['h_d'], user_weights=weights)
        self.assertEqual(len(phi_opt), self.N)
        np.testing.assert_allclose(np.abs(phi_opt), 1.0, atol=1e-6)


class TestFeasibilityAndSelection(unittest.TestCase):
    """Test device selection and feasibility checks under latency/energy budgets."""

    def setUp(self):
        self.config = get_test_config()
        self.channel = WirelessChannel(self.config)
        self.N = self.config['channel']['num_ris_elements']
        self.ris_opt = RISOptimizer(self.N, max_iters=10)

    def test_resource_estimates_positive_and_finite(self):
        selector = JointDeviceRISOptimizer(self.channel, self.ris_opt, self.config)
        state = self.channel.generate_all_channels(5)
        phi = self.ris_opt.random_phases()
        client_data_sizes = [500] * 5
        client_comp = [1e9] * 5
        model_bits = 1_000_000
        tx_w = 0.2

        rates, latencies, energies = selector._resource_estimates(
            state['h_br'], state['h_ru'], state['h_d'], phi,
            client_data_sizes, client_comp, model_bits, tx_w
        )

        self.assertEqual(len(rates), 5)
        self.assertEqual(len(latencies), 5)
        self.assertEqual(len(energies), 5)
        for i in range(5):
            self.assertGreater(rates[i], 0)
            self.assertGreater(latencies[i], 0)
            self.assertGreater(energies[i], 0)
            self.assertTrue(np.isfinite(latencies[i]))
            self.assertTrue(np.isfinite(energies[i]))

    def test_joint_selector_respects_budgets_when_feasible(self):
        cfg = copy.deepcopy(self.config)
        cfg['selection']['latency_budget_ms'] = 500.0  # Generous budget
        cfg['selection']['energy_budget_mj'] = 200.0
        selector = JointDeviceRISOptimizer(self.channel, self.ris_opt, cfg)
        selector.model_size_bits = 1_000_000

        state = self.channel.generate_all_channels(10)
        selected, phi = selector.optimize(
            state['h_br'], state['h_ru'], state['h_d'],
            [200] * 10, [1e9] * 10
        )

        self.assertIsInstance(selected, list)
        self.assertLessEqual(len(selected), 10)
        # Should select at least one when budget is generous
        self.assertGreater(len(selected), 0)

    def test_joint_selector_empty_when_budget_impossible(self):
        """When budget is 0.001 ms, zero clients can be feasible."""
        cfg = copy.deepcopy(self.config)
        cfg['selection']['latency_budget_ms'] = 0.001
        cfg['selection']['energy_budget_mj'] = 0.001
        selector = JointDeviceRISOptimizer(self.channel, self.ris_opt, cfg)
        selector.model_size_bits = 1_000_000

        state = self.channel.generate_all_channels(10)
        selected, phi = selector.optimize(
            state['h_br'], state['h_ru'], state['h_d'],
            [200] * 10, [1e9] * 10
        )
        self.assertEqual(selected, [])

    def test_jain_fairness_calculation(self):
        """Jain's index J = (sum x)^2 / (n * sum x^2)."""
        logger = MetricsLogger("tmp", "test")
        # All equal -> 1.0
        self.assertAlmostEqual(logger.compute_jain_fairness([10.0, 10.0, 10.0]), 1.0)
        # One client gets everything -> 1/n
        self.assertAlmostEqual(logger.compute_jain_fairness([10.0, 0.0, 0.0, 0.0]), 0.25)
        # Empty -> 0.0
        self.assertEqual(logger.compute_jain_fairness([]), 0.0)


class TestCodeReviewFixes(unittest.TestCase):
    """Specific regression tests for findings highlighted in the Code Review Report."""

    def test_iid_partition_allocates_all_samples_no_leftovers(self):
        from fl.data_partition import iid_partition
        from torch.utils.data import TensorDataset
        import torch

        # Test case: 1003 samples across 17 clients (not divisible)
        x = torch.zeros(1003, 1)
        y = torch.zeros(1003, dtype=torch.long)
        ds = TensorDataset(x, y)
        partition = iid_partition(ds, 17)

        all_assigned = []
        for cid, indices in partition.items():
            all_assigned.extend(indices)

        self.assertEqual(len(all_assigned), 1003, "All samples must be assigned without dropping leftovers")
        self.assertEqual(len(set(all_assigned)), 1003, "Indices must be unique across clients")

    def test_dirichlet_noniid_masking_fallback(self):
        from fl.data_partition import dirichlet_noniid_partition
        from torch.utils.data import TensorDataset
        import torch

        # 500 samples across 10 classes, partitioned to 5 clients with high skew
        x = torch.zeros(500, 1)
        y = torch.tensor([i % 10 for i in range(500)], dtype=torch.long)
        ds = TensorDataset(x, y)
        partition = dirichlet_noniid_partition(ds, num_clients=5, alpha=0.1, num_classes=10)

        all_assigned = []
        for cid, indices in partition.items():
            self.assertGreater(len(indices), 0, f"Client {cid} should have non-empty samples")
            all_assigned.extend(indices)
        self.assertEqual(len(all_assigned), 500)

    def test_empty_client_dataset_handling(self):
        from fl.client import FLClient
        from fl.models import get_model
        from torch.utils.data import DataLoader, TensorDataset
        import torch

        x = torch.zeros(0, 3, 32, 32)
        y = torch.zeros(0, dtype=torch.long)
        empty_loader = DataLoader(TensorDataset(x, y), batch_size=32)

        client = FLClient(0, empty_loader, lambda: get_model('cifar10', torch.device('cpu')),
                          {'lr': 0.01, 'momentum': 0.9, 'local_epochs': 1, 'device': 'cpu'})
        model = get_model('cifar10', torch.device('cpu'))
        state = model.state_dict()
        new_state, num_samples, metrics = client.train(state, 0)

        self.assertEqual(num_samples, 0)
        self.assertTrue(metrics.get('empty_dataset', False))
        self.assertTrue(np.isnan(metrics['loss']), "Empty dataset must yield NaN loss, not fake 0.0")

    def test_metrics_logger_diagnostics_and_nan_mean(self):
        import tempfile, json
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(tmpdir, "test_exp")
            # Round 1: normal round with valid measurements
            logger.log_round(1, test_accuracy=50.0, per_round_latency_ms=65.0,
                             mean_client_energy_mj=18.0, idle_round=0,
                             latency_feasible_count=8, energy_feasible_count=9,
                             jointly_feasible_count=8, attempted_upload_count=8,
                             successful_upload_count=7, snr_rejection_count=1,
                             empty_selection=0, aggregation_performed=1,
                             global_model_delta=0.45, latency_is_valid=1, energy_is_valid=1)
            # Round 2: idle round with NaN measurements (not fake zero)
            logger.log_round(2, test_accuracy=50.0, per_round_latency_ms=float('nan'),
                             mean_client_energy_mj=float('nan'), idle_round=1,
                             latency_feasible_count=0, energy_feasible_count=0,
                             jointly_feasible_count=0, attempted_upload_count=0,
                             successful_upload_count=0, snr_rejection_count=0,
                             empty_selection=1, aggregation_performed=0,
                             global_model_delta=0.0, latency_is_valid=0, energy_is_valid=0)

            logger.save_json()
            summary_path = os.path.join(tmpdir, "test_exp_summary.json")
            with open(summary_path, 'r') as f:
                summary = json.load(f)

            # Mean latency should ignore NaN and equal 65.0, not be corrupted by fake 0.0
            self.assertAlmostEqual(summary['mean_latency_ms'], 65.0)
            self.assertAlmostEqual(summary['mean_energy_mj'], 18.0)
            self.assertAlmostEqual(summary['idle_round_fraction'], 0.5)


if __name__ == '__main__':
    unittest.main()
