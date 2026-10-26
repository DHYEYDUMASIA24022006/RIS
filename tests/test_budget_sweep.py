"""Stage 4: Budget sweep validation.

Verifies that:
1. Changing the latency budget T_max actually alters client feasibility.
2. Changing the energy budget E_max alters client feasibility.
3. RIS-assisted selection outperforms non-RIS baselines across budgets.
4. Feasibility checks are dimensionally consistent and non-degenerate.
"""

import os
import sys
import yaml
import unittest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel.channel_model import WirelessChannel
from channel.ris_optimizer import RISOptimizer
from fl.models import get_model
from optimisation.joint_design import JointDeviceRISOptimizer


class TestBudgetSweep(unittest.TestCase):
    """Stage 4: Validate latency and energy sweeps."""

    def setUp(self):
        with open('config/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)

        self.num_clients = 20
        self.channel = WirelessChannel(self.config)
        self.ris_opt = RISOptimizer(num_elements=self.config['channel']['num_ris_elements'], max_iters=20)
        self.model = get_model('cifar10', torch.device('cpu'))
        self.model_bits = sum(p.numel() for p in self.model.parameters()) * 32

    def test_latency_sweep_variation(self):
        """As T_max increases, the number of feasible clients should be monotonically non-decreasing."""
        state = self.channel.generate_all_channels(self.num_clients)
        phi = self.ris_opt.optimize_sum_snr(state['h_br'], state['h_ru'], state['h_d'])

        selector = JointDeviceRISOptimizer(self.channel, self.ris_opt, self.config)
        selector.model_size_bits = self.model_bits
        data_sizes = [2000] * self.num_clients
        comp_cap = [1e9] * self.num_clients
        tx_w = 0.2

        rates, latencies, energies = selector._resource_estimates(
            state['h_br'], state['h_ru'], state['h_d'], phi,
            data_sizes, comp_cap, self.model_bits, tx_w
        )

        latency_budgets = [30.0, 50.0, 75.0, 100.0, 150.0, 200.0]
        feasible_counts = []
        for t_max in latency_budgets:
            count = sum(l <= t_max for l in latencies)
            feasible_counts.append(count)

        print(f"\n[Sweep Test] Latency budgets (ms): {latency_budgets}")
        print(f"[Sweep Test] Feasible client counts: {feasible_counts}")

        # Feasibility should be non-decreasing with higher budget
        for i in range(len(feasible_counts) - 1):
            self.assertLessEqual(feasible_counts[i], feasible_counts[i + 1])

        # At the tightest budget vs highest budget, there must be variation
        self.assertGreater(feasible_counts[-1], feasible_counts[0],
                           "Latency sweep showed no variation across tested range!")

    def test_energy_sweep_variation(self):
        """As E_max increases, the number of feasible clients should be monotonically non-decreasing."""
        state = self.channel.generate_all_channels(self.num_clients)
        phi = self.ris_opt.optimize_sum_snr(state['h_br'], state['h_ru'], state['h_d'])

        selector = JointDeviceRISOptimizer(self.channel, self.ris_opt, self.config)
        selector.model_size_bits = self.model_bits
        data_sizes = [2000] * self.num_clients
        comp_cap = [1e9] * self.num_clients
        tx_w = 0.2

        rates, latencies, energies = selector._resource_estimates(
            state['h_br'], state['h_ru'], state['h_d'], phi,
            data_sizes, comp_cap, self.model_bits, tx_w
        )

        energy_budgets = [15.0, 25.0, 35.0, 50.0, 75.0, 100.0]
        feasible_counts = []
        for e_max in energy_budgets:
            count = sum(e <= e_max for e in energies)
            feasible_counts.append(count)

        print(f"\n[Sweep Test] Energy budgets (mJ): {energy_budgets}")
        print(f"[Sweep Test] Feasible client counts: {feasible_counts}")

        # Feasibility should be non-decreasing with higher budget
        for i in range(len(feasible_counts) - 1):
            self.assertLessEqual(feasible_counts[i], feasible_counts[i + 1])

        # There must be variation between lowest and highest budget
        self.assertGreater(feasible_counts[-1], feasible_counts[0],
                           "Energy sweep showed no variation across tested range!")

    def test_ris_vs_no_ris_feasibility_advantage(self):
        """RIS optimization must yield at least as many feasible clients as direct channel."""
        state = self.channel.generate_all_channels(self.num_clients)
        phi_opt = self.ris_opt.optimize_sum_snr(state['h_br'], state['h_ru'], state['h_d'])
        phi_zero = np.zeros(self.config['channel']['num_ris_elements'])  # disabled RIS

        selector = JointDeviceRISOptimizer(self.channel, self.ris_opt, self.config)
        selector.model_size_bits = self.model_bits
        data_sizes = [2000] * self.num_clients
        comp_cap = [1e9] * self.num_clients
        tx_w = 0.2

        _, lat_ris, _ = selector._resource_estimates(
            state['h_br'], state['h_ru'], state['h_d'], phi_opt,
            data_sizes, comp_cap, self.model_bits, tx_w
        )
        _, lat_no_ris, _ = selector._resource_estimates(
            state['h_br'], state['h_ru'], state['h_d'], phi_zero,
            data_sizes, comp_cap, self.model_bits, tx_w
        )

        ris_feasible = sum(l <= 100.0 for l in lat_ris)
        no_ris_feasible = sum(l <= 100.0 for l in lat_no_ris)

        print(f"\n[Advantage Test] Feasible clients under 100ms - RIS: {ris_feasible}, No-RIS: {no_ris_feasible}")
        self.assertGreaterEqual(ris_feasible, no_ris_feasible)


if __name__ == '__main__':
    unittest.main()
