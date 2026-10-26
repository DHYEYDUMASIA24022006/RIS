"""Feasibility-first, learning- and participation-aware RIS/client scheduler.

This is a transparent greedy alternating-optimization heuristic, not a global
solver. Selection priority combines validation utility, upload debt, and time
since the last successful upload. Hard latency/energy constraints are applied
before ranking and checked again after final RIS optimization.
"""

import numpy as np


class JointDeviceRISOptimizer:
    """Jointly schedule clients and optimize a shared RIS configuration."""

    def __init__(self, channel, ris_optimizer, config: dict):
        self.channel = channel
        self.ris_optimizer = ris_optimizer
        selection = config.get("selection", {})
        fl = config.get("fl", {})
        self.latency_budget = float(selection.get("latency_budget_ms", 100.0))
        self.energy_budget = float(selection.get("energy_budget_mj", 50.0))
        self.max_iters = max(1, int(selection.get("max_alternating_iters", 5)))
        self.tx_power_dbm = float(selection.get("tx_power_dbm", 23.0))
        self.cpu_ghz = float(selection.get("computation_capacity_ghz", 1.0))
        self.cycles_per_sample = float(selection.get("cycles_per_sample", 20.0))
        self.computation_energy_per_sample_mj = float(
            selection.get("computation_energy_per_sample_mj", 0.001))
        self.local_epochs = max(1, int(fl.get("local_epochs", 1)))
        self.num_clients = int(fl.get("num_clients", 20))
        cap = selection.get("max_participants")
        if cap is None:
            cap = int(float(fl.get("fraction_fit", 0.5)) * self.num_clients)
        self.max_clients = max(0, min(self.num_clients, int(cap)))
        self.model_size_bits = selection.get(
            "model_size_bits", config.get("system", {}).get("model_size_bits"))

        # Normalize weights so ablations can set any one of them to zero.
        raw_weights = np.array([
            selection.get("learning_utility_weight", 0.65),
            selection.get("participation_debt_weight", 0.20),
            selection.get("starvation_age_weight", 0.15),
        ], dtype=float)
        if np.any(raw_weights < 0) or not np.isfinite(raw_weights).all():
            raise ValueError("M2 priority weights must be finite and non-negative")
        if raw_weights.sum() <= 0:
            raise ValueError("At least one M2 priority weight must be positive")
        self.utility_weight, self.debt_weight, self.age_weight = raw_weights / raw_weights.sum()
        self.utility_momentum = float(selection.get("utility_ema_momentum", 0.8))
        if not 0.0 <= self.utility_momentum < 1.0:
            raise ValueError("utility_ema_momentum must be in [0, 1)")
        self.max_age = max(1, int(selection.get("starvation_age_cap_rounds", 100)))
        self.utility_scores = np.zeros(self.num_clients, dtype=float)
        self.success_counts = np.zeros(self.num_clients, dtype=float)
        self.selection_counts = np.zeros(self.num_clients, dtype=float)
        self.rounds_since_success = np.zeros(self.num_clients, dtype=float)
        self.rng = np.random.default_rng(int(config.get("system", {}).get("seed", 42)))

    def observe_round(self, selected_indices, successful_indices, utility_gains=None):
        """Update histories after a round; a failed or omitted client ages."""
        selected = {int(i) for i in selected_indices}
        successful = {int(i) for i in successful_indices}
        if any(i < 0 or i >= self.num_clients for i in selected | successful):
            raise IndexError("Client index outside configured client range")
        self.selection_counts[list(selected)] += 1.0
        self.rounds_since_success = np.minimum(
            self.rounds_since_success + 1.0, self.max_age)
        if successful:
            ids = np.fromiter(successful, dtype=int)
            self.success_counts[ids] += 1.0
            self.rounds_since_success[ids] = 0.0
        gains = utility_gains or {}
        for idx in successful:
            if idx in gains:
                gain = float(gains[idx])
                if np.isfinite(gain):
                    self.utility_scores[idx] = (
                        self.utility_momentum * self.utility_scores[idx]
                        + (1.0 - self.utility_momentum) * gain)

    def optimize(self, h_br, h_ru_list, h_d_list, client_data_sizes,
                 client_comp_capacities):
        """Return up to K feasible client indices and final RIS phases."""
        n = len(h_ru_list)
        expected = (len(h_d_list), len(client_data_sizes), len(client_comp_capacities))
        if any(size != n for size in expected):
            raise ValueError("Channel, data-size, and compute-capacity lists must match")
        if n == 0 or self.max_clients == 0:
            return [], self.ris_optimizer.random_phases()
        if n > self.num_clients:
            raise ValueError("Runtime client count exceeds configured fl.num_clients")

        model_bits = int(self.model_size_bits or 1_000_000)
        if model_bits <= 0:
            raise ValueError("model_size_bits must be positive")
        tx_w = 10.0 ** ((self.tx_power_dbm - 30.0) / 10.0)
        priorities = self._priorities(n)
        # A seeded random tie key avoids systematic preference for low client IDs.
        tie_key = self.rng.random(n)
        phi = self.ris_optimizer.optimize_sum_snr(
            h_br, h_ru_list, h_d_list, user_weights=priorities)
        selected = []

        for _ in range(self.max_iters):
            _, latency, energy = self._resource_estimates(
                h_br, h_ru_list, h_d_list, phi, client_data_sizes,
                client_comp_capacities, model_bits, tx_w)
            feasible = [i for i in range(n)
                        if latency[i] <= self.latency_budget
                        and energy[i] <= self.energy_budget]
            updated = self._greedy_feasible_selection(feasible, priorities, tie_key)
            if updated == selected and selected:
                break
            selected = updated
            if not selected:
                return [], phi
            phi = self.ris_optimizer.optimize_sum_snr(
                h_br, [h_ru_list[i] for i in selected],
                [h_d_list[i] for i in selected], user_weights=priorities[selected])

        # Final RIS phases can change feasibility. Refill vacated slots from
        # remaining feasible clients, ranked by the same priority, then recheck.
        _, latency, energy = self._resource_estimates(
            h_br, h_ru_list, h_d_list, phi, client_data_sizes,
            client_comp_capacities, model_bits, tx_w)
        selected = [i for i in selected
                    if latency[i] <= self.latency_budget
                    and energy[i] <= self.energy_budget]
        feasible = [i for i in range(n)
                    if latency[i] <= self.latency_budget
                    and energy[i] <= self.energy_budget and i not in selected]
        self.last_diagnostics = {
            'latency_feasible_count': sum(1 for i in range(n) if latency[i] <= self.latency_budget),
            'energy_feasible_count': sum(1 for i in range(n) if energy[i] <= self.energy_budget),
            'jointly_feasible_count': sum(1 for i in range(n) if latency[i] <= self.latency_budget and energy[i] <= self.energy_budget),
            'selected_count': len(selected),
            'empty_selection': len(selected) == 0,
        }
        return selected, phi

    def _priorities(self, n):
        utility = self._rank_scale(self.utility_scores[:n])
        debt = self._rank_scale(-self.success_counts[:n])
        age = self._rank_scale(self.rounds_since_success[:n])
        return (self.utility_weight * utility + self.debt_weight * debt
                + self.age_weight * age)

    @staticmethod
    def _rank_scale(values):
        values = np.asarray(values, dtype=float)
        if values.size <= 1 or np.allclose(values, values[0]):
            return np.ones(values.size, dtype=float)
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        ranks = np.empty(values.size, dtype=float)
        start = 0
        while start < values.size:
            end = start + 1
            while end < values.size and sorted_values[end] == sorted_values[start]:
                end += 1
            ranks[order[start:end]] = (start + end - 1) / 2.0
            start = end
        return ranks / max(values.size - 1, 1)

    def _resource_estimates(self, h_br, h_ru_list, h_d_list, phi,
                            client_data_sizes, client_comp_capacities,
                            model_bits, tx_w):
        rates, latencies, energies = [], [], []
        for k in range(len(h_ru_list)):
            h_eff = self.channel.compute_effective_channel(
                h_br, h_ru_list[k], h_d_list[k], phi)
            snr = float(np.real(self.channel.compute_snr(h_eff, self.tx_power_dbm)))
            rate = max(float(self.channel.compute_achievable_rate(max(snr, 0.0))), 1e-12)
            size = max(0, int(client_data_sizes[k]))
            capacity = float(client_comp_capacities[k])
            cpu_hz = capacity if capacity > 1e6 else self.cpu_ghz * 1e9
            compute_s = size * self.local_epochs * self.cycles_per_sample / max(cpu_hz, 1e-12)
            comm_s = model_bits / rate
            rates.append(rate)
            latencies.append((compute_s + comm_s) * 1000.0)
            compute_mj = size * self.local_epochs * self.computation_energy_per_sample_mj
            energies.append(tx_w * comm_s * 1000.0 + compute_mj)
        return rates, latencies, energies

    def _greedy_feasible_selection(self, feasible, priorities, tie_key, slots=None):
        slots = self.max_clients if slots is None else max(0, int(slots))
        ranked = sorted(feasible, key=lambda i: (
            priorities[i], -self.success_counts[i],
            self.rounds_since_success[i], tie_key[i]), reverse=True)
        return [int(i) for i in ranked[:slots]]
