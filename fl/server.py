"""Federated Learning server with wireless-aware aggregation.

Integrates:
  - Wireless channel model (SNR-based uplink success/failure)
  - RIS phase shift optimisation
  - Device selection strategies
  - FedAvg and AirComp aggregation
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .client import FLClient


class FLServer:
    """FL server with wireless-aware aggregation."""

    def __init__(self, model_fn, test_loader, config: dict,
                 channel=None, ris_optimizer=None, device_selector=None,
                 aircomp_aggregator=None, validation_loader=None):
        """
        Args:
            model_fn: Callable returning a fresh nn.Module.
            test_loader: DataLoader for test set.
            config: Full config dict.
            channel: WirelessChannel instance (None = ideal channel).
            ris_optimizer: RISOptimizer instance (None = no RIS).
            device_selector: JointDeviceRISOptimizer or baseline selector.
            aircomp_aggregator: AirCompAggregator instance (None = use FedAvg).
        """
        requested_device = config.get('system', {}).get('device', 'auto')
        if requested_device == 'auto':
            requested_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(requested_device)
        self.model_fn = model_fn
        self.global_model = model_fn().to(self.device)
        self.test_loader = test_loader
        self.validation_loader = validation_loader
        self.config = config
        seed = int(config.get('system', {}).get('seed', 42))
        self._channel_rng = np.random.RandomState(seed)

        # Wireless components
        self.channel = channel
        self.ris_optimizer = ris_optimizer
        self.device_selector = device_selector
        self.aircomp = aircomp_aggregator

        # FL config
        fl_cfg = config.get('fl', {})
        self.fraction_fit = fl_cfg.get('fraction_fit', 0.5)
        self.target_accuracy = fl_cfg.get('target_accuracy', 0.90) * 100  # %

        # Selection config
        sel_cfg = config.get('selection', {})
        self.tx_power_dbm = sel_cfg.get('tx_power_dbm', 23.0)
        self.selection_strategy = sel_cfg.get('strategy', 'random')
        self.latency_budget_ms = sel_cfg.get('latency_budget_ms', 100.0)
        self.energy_budget_mj = sel_cfg.get('energy_budget_mj', 50.0)
        self.cpu_ghz = sel_cfg.get('computation_capacity_ghz', 1.0)
        self.cycles_per_sample = sel_cfg.get('cycles_per_sample', 20.0)
        self.computation_energy_per_sample_mj = sel_cfg.get(
            'computation_energy_per_sample_mj', 0.001)
        self.local_epochs = fl_cfg.get('local_epochs', 1)

        # Wireless impairment: outage + noisy uploads (must affect aggregation)
        self.snr_threshold_dB = float(fl_cfg.get('snr_outage_threshold_db', 5.0))
        self.use_rate_outage = bool(fl_cfg.get('use_rate_outage', True))
        self.update_noise_scale = float(fl_cfg.get('update_noise_scale', 0.05))
        self.enforce_energy_latency = bool(fl_cfg.get('enforce_energy_latency', False))
        self.max_participants = sel_cfg.get('max_participants')

        # Analog aggregation configuration.  AirComp is deliberately a server
        # concern because it aggregates *model updates*, not just synthetic
        # vectors in a separate experiment.
        air_cfg = config.get('aircomp', {})
        self.aircomp_max_power_w = 10 ** ((float(
            air_cfg.get('max_tx_power_dbm', self.tx_power_dbm)) - 30) / 10.0)
        self.aircomp_normalise_updates = bool(air_cfg.get('normalise_updates', True))

    def run(self, clients: list, num_rounds: int,
            metrics_logger=None, use_ris: bool = True,
            ris_mode: str = 'optimized', verbose: bool = True) -> dict:
        """Main FL training loop with wireless simulation.

        Args:
            clients: List of FLClient instances.
            num_rounds: Number of communication rounds.
            metrics_logger: MetricsLogger instance (optional).
            use_ris: Whether to use RIS.
            ris_mode: 'optimized', 'random', or 'none'.
            verbose: Print progress.

        Returns:
            dict with lists: accuracy, loss, snr, outage, latency, etc.
        """
        num_clients = len(clients)
        results = {
            'accuracy': [], 'loss': [], 'round': [],
            'num_participating': [], 'num_successful': [],
            'mean_snr_db': [], 'min_snr_db': [], 'max_snr_db': [],
            'outage_prob': [], 'per_round_latency_ms': [],
            'mean_energy_mj': [], 'aggregation_mse': [],
            'mean_uplink_rate_bps': [],
            'selection_fairness': [], 'successful_upload_fairness': [],
            'client_selection_counts': [], 'client_success_counts': [],
            'rounds_to_target': [], 'idle_round': [],
            'resource_constraint_violations': [],
            'latency_feasible_count': [], 'energy_feasible_count': [],
            'jointly_feasible_count': [], 'attempted_upload_count': [],
            'snr_rejection_count': [], 'empty_selection': [],
            'aggregation_performed': [], 'global_model_delta': [],
            'latency_is_valid': [], 'energy_is_valid': [],
        }
        selection_counts = np.zeros(num_clients, dtype=int)
        success_counts = np.zeros(num_clients, dtype=int)
        first_target_round = None

        iterator = range(num_rounds)
        if verbose:
            iterator = tqdm(iterator, desc='FL Training')

        snr_threshold_lin = 10 ** (self.snr_threshold_dB / 10.0)
        model_bits = self.get_model_size_bits()
        tx_power_w = 10 ** ((self.tx_power_dbm - 30) / 10.0)

        for round_idx in iterator:
            # ----- 1. Generate channel state ----- #
            snr_list = []
            rate_list = []
            aircomp_gains = []
            phi = None
            h_br = h_ru = h_d = None

            if self.channel is not None:
                ch_state = self._generate_channel_state(num_clients)
                h_br = ch_state['h_br']
                h_ru = ch_state['h_ru']
                h_d = ch_state['h_d']

                # ----- 2. RIS phases (joint selector may override) ----- #
                if use_ris and self.ris_optimizer is not None:
                    if ris_mode == 'optimized':
                        phi = self.ris_optimizer.optimize_sum_snr(h_br, h_ru, h_d)
                    elif ris_mode == 'aircomp':
                        phi = self.ris_optimizer.optimize_for_aircomp(h_br, h_ru, h_d)
                    elif ris_mode == 'random':
                        phi = self.ris_optimizer.random_phases()
                    else:
                        phi = None

                if (self.device_selector is not None
                        and hasattr(self.device_selector, 'optimize')):
                    if hasattr(self.device_selector, 'model_size_bits'):
                        self.device_selector.model_size_bits = model_bits
                    data_sizes = [len(c.dataloader.dataset) for c in clients]
                    cpu_hz = [self.cpu_ghz * 1e9] * num_clients
                    selected_idx, phi_joint = self.device_selector.optimize(
                        h_br, h_ru, h_d, data_sizes, cpu_hz)
                    if use_ris:
                        phi = phi_joint
                else:
                    selected_idx = None

                for k in range(num_clients):
                    if phi is not None:
                        h_eff = self.channel.compute_effective_channel(
                            h_br, h_ru[k], h_d[k], phi)
                    else:
                        h_eff = h_d[k]
                    snr_lin = self.channel.compute_snr(h_eff, self.tx_power_dbm)
                    snr_list.append(float(np.real(snr_lin)))
                    rate_list.append(self.channel.compute_achievable_rate(snr_list[-1]))
                    # MRC gives a real, non-negative scalar equivalent gain.
                    aircomp_gains.append(complex(np.linalg.norm(h_eff)))
            else:
                snr_list = [1e10] * num_clients
                rate_list = [1e12] * num_clients
                aircomp_gains = [1.0 + 0j] * num_clients
                selected_idx = None

            # ----- 3. Select who is scheduled to transmit ----- #
            default_m = max(1, int(self.fraction_fit * num_clients))
            m = min(num_clients, int(self.max_participants or default_m))
            resource_values = [self._client_resource_values(
                len(clients[k].dataloader.dataset), rate_list[k], model_bits)
                for k in range(num_clients)]
            latency_feasible_count = sum(1 for (t_ms, _) in resource_values if t_ms <= self.latency_budget_ms)
            energy_feasible_count = sum(1 for (_, e_mj) in resource_values if e_mj <= self.energy_budget_mj)
            feasible_indices = [k for k, (latency_ms, energy_mj) in enumerate(resource_values)
                                if (not self.enforce_energy_latency
                                    or (latency_ms <= self.latency_budget_ms
                                        and energy_mj <= self.energy_budget_mj))]
            jointly_feasible_count = len(feasible_indices)
            if selected_idx is None:
                if self.selection_strategy == 'random':
                    count = min(m, len(feasible_indices))
                    selected_idx = (list(np.random.choice(feasible_indices, count, replace=False))
                                    if count else [])
                elif self.selection_strategy in ('channel_only', 'snr_top',
                                                  'channel_random_ris'):
                    selected_idx = sorted(feasible_indices,
                                          key=lambda k: snr_list[k], reverse=True)[:m]
                elif self.device_selector is not None and hasattr(self.device_selector, 'select'):
                    selected_idx = [int(k) for k in self.device_selector.select(snr_list, m)
                                    if int(k) in feasible_indices][:m]
                elif self.fraction_fit >= 0.999:
                    selected_idx = [k for k in range(num_clients)
                                    if k in feasible_indices]
                    selected_idx = selected_idx[:m]
                else:
                    count = min(m, len(feasible_indices))
                    selected_idx = (list(np.random.choice(feasible_indices, count, replace=False))
                                    if count else [])
            else:
                selected_idx = [int(k) for k in selected_idx
                                if int(k) in feasible_indices]

            for idx in selected_idx:
                selection_counts[idx] += 1

            # ----- 4. Outage: dropped updates (channel actually filters FL) ----- #
            successful_global_idx = []
            latencies_ms = []
            energies_mj = []
            attempted_upload_count = len(selected_idx)
            snr_rejection_count = 0
            for idx in selected_idx:
                t_ms, e_mj = resource_values[idx]
                latencies_ms.append(t_ms)
                energies_mj.append(e_mj)

                snr_ok = snr_list[idx] >= snr_threshold_lin
                if not snr_ok:
                    snr_rejection_count += 1
                rate_ok = (not self.use_rate_outage) or (t_ms <= self.latency_budget_ms)
                energy_ok = (not self.enforce_energy_latency) or (e_mj <= self.energy_budget_mj)
                if self.channel is None:
                    successful_global_idx.append(idx)
                elif snr_ok and rate_ok and energy_ok:
                    successful_global_idx.append(idx)

            for idx in successful_global_idx:
                success_counts[idx] += 1

            # ----- 5–6. Local training + SNR-dependent uplink noise ----- #
            global_params = copy.deepcopy(self.global_model.state_dict())
            client_updates = []
            mse_list = []
            utility_gains = {}
            track_utility = (self.validation_loader is not None
                             and self.device_selector is not None
                             and hasattr(self.device_selector, 'observe_round'))
            baseline_validation_loss = (
                self._validation_loss_for_state(global_params)
                if track_utility else None)

            for idx in successful_global_idx:
                local_params, n_samples, _ = clients[idx].train(
                    copy.deepcopy(global_params), round_idx=round_idx)
                if baseline_validation_loss is not None:
                    candidate_loss = self._validation_loss_for_state(local_params)
                    utility_gains[idx] = baseline_validation_loss - candidate_loss
                if (self.aircomp is None and self.channel is not None
                        and self.update_noise_scale > 0):
                    noisy, mse = self._impair_upload(local_params, snr_list[idx])
                    mse_list.append(mse)
                    client_updates.append((noisy, n_samples))
                else:
                    mse_list.append(0.0)
                    client_updates.append((local_params, n_samples))

            # ----- 7. Aggregate ----- #
            aggregation_performed = int(len(client_updates) > 0)
            if len(client_updates) > 0:
                if self.aircomp is not None and self.channel is not None:
                    gains = [aircomp_gains[idx] for idx in successful_global_idx]
                    aggregated, aircomp_mse = self.aircomp_aggregate(
                        client_updates, global_params, gains)
                    mse_list = [aircomp_mse]
                else:
                    aggregated = self.fedavg_aggregate(client_updates)
                self.global_model.load_state_dict(aggregated)
                param_delta = sum(torch.sum((self.global_model.state_dict()[k].float() - global_params[k].float())**2).item()
                                  for k in global_params if global_params[k].is_floating_point())
                global_model_delta = float(np.sqrt(param_delta))
            else:
                global_model_delta = 0.0

            if self.device_selector is not None and hasattr(self.device_selector, 'observe_round'):
                self.device_selector.observe_round(
                    selected_idx, successful_global_idx, utility_gains)

            # ----- 8. Evaluate ----- #
            loss, accuracy = self.evaluate()
            if first_target_round is None and accuracy >= self.target_accuracy:
                first_target_round = round_idx + 1

            # ----- 9. Metrics (SNR of scheduled clients, not all K) ----- #
            snr_db_all = [10 * np.log10(max(s, 1e-20)) for s in snr_list]
            if selected_idx:
                snr_db_sel = [10 * np.log10(max(snr_list[i], 1e-20)) for i in selected_idx]
                rates_sel = [rate_list[i] for i in selected_idx]
            else:
                snr_db_sel = snr_db_all
                rates_sel = rate_list

            mean_snr_db = float(np.mean(snr_db_sel)) if snr_db_sel else 0.0
            min_snr_db = float(np.min(snr_db_sel)) if snr_db_sel else 0.0
            max_snr_db = float(np.max(snr_db_sel)) if snr_db_sel else 0.0
            outage = (1.0 - len(successful_global_idx) / len(selected_idx)
                      if selected_idx else 0.0)
            agg_mse = float(np.mean(mse_list)) if mse_list else 0.0
            mean_rate = float(np.mean(rates_sel)) if rates_sel else 0.0
            latency_is_valid = int(len(latencies_ms) > 0)
            energy_is_valid = int(len(energies_mj) > 0)
            max_latency_ms = float(np.max(latencies_ms)) if latencies_ms else float('nan')
            mean_energy = float(np.mean(energies_mj)) if energies_mj else float('nan')
            selection_fairness = self.jain_fairness(selection_counts)
            success_fairness = self.jain_fairness(success_counts)
            violations = sum(
                int(latencies_ms[i] > self.latency_budget_ms
                    or energies_mj[i] > self.energy_budget_mj)
                for i in range(len(selected_idx))) if selected_idx else 0

            results['accuracy'].append(accuracy)
            results['loss'].append(loss)
            results['round'].append(round_idx + 1)
            results['num_participating'].append(len(selected_idx))
            results['num_successful'].append(len(successful_global_idx))
            results['mean_snr_db'].append(mean_snr_db)
            results['min_snr_db'].append(min_snr_db)
            results['max_snr_db'].append(max_snr_db)
            results['outage_prob'].append(outage)
            results['per_round_latency_ms'].append(max_latency_ms)
            results['mean_energy_mj'].append(mean_energy)
            results['aggregation_mse'].append(agg_mse)
            results['mean_uplink_rate_bps'].append(mean_rate)
            results['selection_fairness'].append(selection_fairness)
            results['successful_upload_fairness'].append(success_fairness)
            results['client_selection_counts'].append(selection_counts.tolist())
            results['client_success_counts'].append(success_counts.tolist())
            results['idle_round'].append(int(not selected_idx))
            results['resource_constraint_violations'].append(violations)
            results['rounds_to_target'].append(
                first_target_round if first_target_round is not None else -1)
            results['latency_feasible_count'].append(latency_feasible_count)
            results['energy_feasible_count'].append(energy_feasible_count)
            results['jointly_feasible_count'].append(jointly_feasible_count)
            results['attempted_upload_count'].append(attempted_upload_count)
            results['snr_rejection_count'].append(snr_rejection_count)
            results['empty_selection'].append(int(not selected_idx))
            results['aggregation_performed'].append(aggregation_performed)
            results['global_model_delta'].append(global_model_delta)
            results['latency_is_valid'].append(latency_is_valid)
            results['energy_is_valid'].append(energy_is_valid)

            if verbose:
                iterator.set_postfix({
                    'acc': f'{accuracy:.1f}%',
                    'ok': f'{len(successful_global_idx)}/{len(selected_idx)}',
                    'SNR': f'{mean_snr_db:.1f}dB',
                    'out': f'{outage:.2f}',
                })

            if metrics_logger is not None:
                metrics_logger.log_round(
                    round_num=round_idx + 1,
                    test_accuracy=accuracy,
                    test_loss=loss,
                    num_participating_clients=len(selected_idx),
                    num_successful_uploads=len(successful_global_idx),
                    aggregation_mse=agg_mse,
                    mean_snr_db=mean_snr_db,
                    min_snr_db=min_snr_db,
                    max_snr_db=max_snr_db,
                    outage_probability=outage,
                    mean_uplink_rate_bps=mean_rate,
                    per_round_latency_ms=max_latency_ms if latency_is_valid else None,
                    mean_client_energy_mj=mean_energy if energy_is_valid else None,
                    jain_fairness_index=success_fairness,
                    selection_strategy=self.selection_strategy,
                    rounds_to_target=first_target_round if first_target_round is not None else -1,
                    idle_round=int(not selected_idx),
                    resource_constraint_violations=violations,
                    latency_feasible_count=latency_feasible_count,
                    energy_feasible_count=energy_feasible_count,
                    jointly_feasible_count=jointly_feasible_count,
                    attempted_upload_count=attempted_upload_count,
                    snr_rejection_count=snr_rejection_count,
                    empty_selection=int(not selected_idx),
                    aggregation_performed=aggregation_performed,
                    global_model_delta=global_model_delta,
                    latency_is_valid=latency_is_valid,
                    energy_is_valid=energy_is_valid,
                )

        return results

    def _client_resource_values(self, n_samples: int, rate_bps: float,
                                model_bits: int) -> tuple[float, float]:
        """Return estimated client completion time (ms) and energy (mJ)."""
        compute_s = (n_samples * self.local_epochs * self.cycles_per_sample
                     / max(self.cpu_ghz * 1e9, 1e-12))
        comm_s = model_bits / max(float(rate_bps), 1e-12)
        tx_power_w = 10.0 ** ((float(self.tx_power_dbm) - 30.0) / 10.0)
        latency_ms = (compute_s + comm_s) * 1000.0
        energy_mj = (tx_power_w * comm_s * 1000.0
                     + n_samples * self.local_epochs
                     * self.computation_energy_per_sample_mj)
        return float(latency_ms), float(energy_mj)

    def _generate_channel_state(self, num_clients: int) -> dict:
        """Use an isolated channel RNG so methods see paired channel traces."""
        global_state = np.random.get_state()
        np.random.set_state(self._channel_rng.get_state())
        try:
            state = self.channel.generate_all_channels(num_clients)
            self._channel_rng.set_state(np.random.get_state())
        finally:
            np.random.set_state(global_state)
        return state

    def _validation_loss_for_state(self, state_dict) -> float:
        """Evaluate one candidate client model on the held-out validation set."""
        if not hasattr(self, '_val_model') or self._val_model is None:
            self._val_model = self.model_fn().to(self.device)
        self._val_model.load_state_dict(state_dict)
        self._val_model.eval()
        criterion = nn.CrossEntropyLoss(reduction='sum')
        total_loss = 0.0
        total = 0
        if not hasattr(self, '_cached_val_batches') or self._cached_val_batches is None:
            self._cached_val_batches = [
                (data.to(self.device), target.to(self.device))
                for data, target in self.validation_loader
            ] if self.validation_loader is not None else []
        with torch.no_grad():
            for data, target in self._cached_val_batches:
                total_loss += float(criterion(self._val_model(data), target).item())
                total += int(data.size(0))
        return total_loss / max(total, 1)

    @staticmethod
    def jain_fairness(values: np.ndarray) -> float:
        """Jain's index for actual cumulative scheduling or upload counts."""
        values = np.asarray(values, dtype=float)
        denominator = len(values) * np.sum(values ** 2)
        return float(np.sum(values) ** 2 / denominator) if denominator > 0 else 0.0

    def aircomp_aggregate(self, client_updates: list[tuple[dict, int]],
                          global_params: dict, channel_gains: list[complex]
                          ) -> tuple[dict, float]:
        """Aggregate client model deltas through a power-constrained AirComp MAC.

        The update vectors are normalised jointly before transmission so that
        the transmit-power model applies to symbols rather than arbitrary model
        units.  The returned MSE is against exact sample-weighted FedAvg.
        """
        float_keys = [key for key, tensor in global_params.items()
                      if tensor.is_floating_point()]
        shapes = [global_params[key].shape for key in float_keys]
        sizes = [global_params[key].numel() for key in float_keys]

        deltas = []
        sample_counts = []
        for params, num_samples in client_updates:
            flat = torch.cat([
                (params[key].detach().to('cpu', dtype=torch.float32)
                 - global_params[key].detach().to('cpu', dtype=torch.float32)).reshape(-1)
                for key in float_keys
            ]).numpy()
            deltas.append(flat)
            sample_counts.append(num_samples)

        weights = np.asarray(sample_counts, dtype=float)
        weights /= np.sum(weights)
        exact_delta = np.sum(np.asarray(deltas) * weights[:, None], axis=0)
        scale = max(float(np.max(np.abs(np.asarray(deltas)))), 1e-12)
        transmitted = [delta / scale for delta in deltas] if self.aircomp_normalise_updates else deltas
        estimated = self.aircomp.aggregate(
            transmitted, channel_gains, self.channel.noise_power_w,
            self.aircomp_max_power_w, weights.tolist())
        if self.aircomp_normalise_updates:
            estimated *= scale

        mse = self.aircomp.compute_aggregation_mse(exact_delta, estimated)
        aggregated = copy.deepcopy(global_params)
        offset = 0
        for key, shape, size in zip(float_keys, shapes, sizes):
            update = torch.from_numpy(estimated[offset:offset + size].reshape(shape))
            aggregated[key] = (global_params[key].detach().to('cpu', dtype=torch.float32)
                               + update).to(dtype=global_params[key].dtype,
                                            device=global_params[key].device)
            offset += size
        return aggregated, mse

    def _impair_upload(self, state_dict: dict, snr_lin: float) -> tuple[dict, float]:
        """Corrupt uploaded weights with SNR-dependent AWGN (digital uplink).

        Variance scales as 1/SNR so a weak link distorts FedAvg, not just a log.
        """
        noisy = copy.deepcopy(state_dict)
        sigma = self.update_noise_scale / np.sqrt(max(float(np.real(snr_lin)), 1e-12))
        se, n = 0.0, 0
        for key, tensor in noisy.items():
            if not tensor.is_floating_point():
                continue
            noise = torch.randn_like(tensor) * sigma
            se += float(torch.sum(noise.float() ** 2).item())
            n += tensor.numel()
            noisy[key] = tensor + noise.to(dtype=tensor.dtype)
        mse = se / max(n, 1)
        return noisy, mse

    def fedavg_aggregate(self, client_updates: list[tuple[dict, int]]) -> dict:
        """FedAvg: weighted average of client parameters.

        w_global = sum(n_k * w_k) / sum(n_k)
        """
        total_samples = sum(n for _, n in client_updates)
        if total_samples == 0:
            return self.global_model.state_dict()

        aggregated = copy.deepcopy(client_updates[0][0])
        weight = client_updates[0][1] / total_samples

        for key in aggregated:
            if aggregated[key].is_floating_point():
                aggregated[key] = aggregated[key].float() * weight

        for params, n_samples in client_updates[1:]:
            w = n_samples / total_samples
            for key in aggregated:
                if aggregated[key].is_floating_point():
                    aggregated[key] += params[key].float() * w

        return aggregated

    def evaluate(self) -> tuple[float, float]:
        """Evaluate global model on test set.

        Returns:
            (loss, accuracy_percent)
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                total_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)

        avg_loss = total_loss / max(total, 1)
        accuracy = 100.0 * correct / max(total, 1)
        return avg_loss, accuracy

    def get_model_size_bits(self) -> int:
        """Model size in bits (parameters * 32)."""
        return sum(p.numel() for p in self.global_model.parameters()) * 32

    def reset_model(self):
        """Reset global model to random initialisation."""
        self.global_model = self.model_fn().to(self.device)
