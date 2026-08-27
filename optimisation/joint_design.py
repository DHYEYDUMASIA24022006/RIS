"""Joint design of device selection and RIS phase shifts."""

import numpy as np


class JointDeviceRISOptimizer:
    """Joint optimisation of client selection and RIS phase shifts.

    Alternating optimisation:
      1. Fix RIS phases, keep clients that meet T_max and E_max.
      2. Re-optimise RIS (sum-SNR) over the selected set.
      Repeat until the selected set is stable.
    """

    def __init__(self, channel, ris_optimizer, config: dict):
        self.channel = channel
        self.ris_optimizer = ris_optimizer

        selection_cfg = config.get('selection', {})
        self.latency_budget = selection_cfg.get('latency_budget_ms', 100.0)
        self.energy_budget = selection_cfg.get('energy_budget_mj', 50.0)
        self.max_iters = selection_cfg.get('max_alternating_iters', 5)
        self.tx_power_dbm = selection_cfg.get('tx_power_dbm', 23.0)
        self.cpu_ghz = selection_cfg.get('computation_capacity_ghz', 1.0)
        self.cycles_per_sample = selection_cfg.get('cycles_per_sample', 20.0)
        self.computation_energy_per_sample_mj = selection_cfg.get(
            'computation_energy_per_sample_mj', 0.001)
        self.local_epochs = config.get('fl', {}).get('local_epochs', 1)
        self.max_clients = selection_cfg.get('max_participants', None)
        if self.max_clients is None:
            fl_cfg = config.get('fl', {})
            self.max_clients = max(1, int(fl_cfg.get('fraction_fit', 0.5)
                                          * fl_cfg.get('num_clients', 20)))

        self.model_size_bits = config.get('system', {}).get('model_size_bits', None)

    def optimize(self, h_br: np.ndarray, h_ru_list: list[np.ndarray],
                 h_d_list: list[np.ndarray],
                 client_data_sizes: list[int],
                 client_comp_capacities: list[float]) -> tuple[list[int], np.ndarray]:
        """Returns (selected client indices, complex RIS phase vector)."""
        num_clients = len(h_ru_list)
        phi = self.ris_optimizer.optimize_sum_snr(h_br, h_ru_list, h_d_list)
        selected_clients = list(range(num_clients))
        model_bits = self.model_size_bits or 1_000_000
        tx_w = 10 ** ((self.tx_power_dbm - 30) / 10.0)

        for iteration in range(self.max_iters):
            rates, latencies, energies = [], [], []
            for k in range(num_clients):
                h_eff = self.channel.compute_effective_channel(
                    h_br, h_ru_list[k], h_d_list[k], phi)
                snr = float(np.real(self.channel.compute_snr(h_eff, self.tx_power_dbm)))
                rate = self.channel.compute_achievable_rate(max(snr, 0.0))
                rates.append(rate)
                latencies.append(self._compute_client_latency(
                    rate, client_data_sizes[k], client_comp_capacities[k], model_bits))
                comm_time_s = model_bits / (rate + 1e-12)
                energies.append(self._compute_client_energy(
                    tx_w, comm_time_s, client_data_sizes[k]))

            new_selected = self._greedy_selection(rates, latencies, energies)
            if not new_selected:
                new_selected = [int(np.argmax(rates))]
            if set(new_selected) == set(selected_clients) and iteration > 0:
                selected_clients = new_selected
                break

            selected_clients = new_selected

            h_ru_sel = [h_ru_list[i] for i in selected_clients]
            h_d_sel = [h_d_list[i] for i in selected_clients]
            phi = self.ris_optimizer.optimize_sum_snr(h_br, h_ru_sel, h_d_sel)

        if not selected_clients:
            selected_clients = [int(np.argmax(rates))] if rates else [0]
        return selected_clients, phi

    def _compute_client_latency(self, rate_bps: float, data_size: int,
                                comp_capacity: float, model_bits: int) -> float:
        cpu_hz = comp_capacity if comp_capacity > 1e6 else self.cpu_ghz * 1e9
        comp_time_s = (data_size * self.local_epochs * self.cycles_per_sample) / (cpu_hz + 1e-12)
        comm_time_s = model_bits / (rate_bps + 1e-12)
        return (comp_time_s + comm_time_s) * 1000.0

    def _compute_client_energy(self, tx_power_w: float, comm_time_s: float,
                               data_size: int) -> float:
        comm_energy_mj = tx_power_w * comm_time_s * 1000.0
        comp_energy_mj = data_size * self.local_epochs * self.computation_energy_per_sample_mj
        return comm_energy_mj + comp_energy_mj

    def _greedy_selection(self, rates, latencies, energies) -> list[int]:
        sorted_indices = np.argsort(rates)[::-1]
        selected = []
        for idx in sorted_indices:
            if latencies[idx] <= self.latency_budget and energies[idx] <= self.energy_budget:
                selected.append(int(idx))
                if len(selected) >= self.max_clients:
                    break
        return selected
