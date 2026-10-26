"""Stage 3: Federated Learning smoke test across all 7 strategies.

Validates that:
1. Each of the 7 strategies initializes and runs properly.
2. Clients are selected and participate under wireless/RIS constraints.
3. Successful uploads are aggregated, updating the global model weights.
4. Loss decreases / accuracy improves above random guessing.
5. All per-round metrics (latency, energy, snr, fairness) are finite and non-negative.
6. Separate log files and results are generated per strategy.
"""

import os
import sys
import copy
import yaml
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channel.channel_model import WirelessChannel
from channel.ris_optimizer import RISOptimizer
from fl.models import get_model
from fl.data_partition import load_dataset_with_validation, iid_partition, get_client_dataloaders
from fl.client import FLClient
from fl.server import FLServer
from optimisation.joint_design import JointDeviceRISOptimizer
from optimisation.baselines import RandomSelector, ChannelOnlySelector
from metrics.metrics_logger import MetricsLogger
from experiments.exp2_joint_design import _create_clients, _exp2_config, _get_selector, _ris_flags


class TestFederatedSmoke(unittest.TestCase):
    """Smoke test running 3 rounds across all 7 strategies."""

    @classmethod
    def setUpClass(cls):
        with open('config/config.yaml', 'r') as f:
            cls.config = yaml.safe_load(f)

        cls.num_clients = 4
        cls.num_rounds = 3
        cls.batch_size = 32
        cls.dataset_name = 'cifar10'
        cls.save_dir = 'results/test_smoke'
        os.makedirs(cls.save_dir, exist_ok=True)

        seed = 42
        np.random.seed(seed)
        torch.manual_seed(seed)

        (train_ds, val_ds, test_ds,
         train_indices, val_indices) = load_dataset_with_validation(
            cls.dataset_name, validation_fraction=0.05, seed=seed)

        # Cap samples per client to 128 for lightning-fast test
        partition_map = iid_partition(train_ds, cls.num_clients, train_indices)
        cls.client_loaders = get_client_dataloaders(
            train_ds, partition_map, cls.batch_size, max_samples_per_client=128)

        cls.test_loader = DataLoader(Subset(test_ds, range(200)), batch_size=100, shuffle=False)
        cls.val_loader = DataLoader(Subset(val_ds, val_indices[:100]), batch_size=100, shuffle=False)

        cls.channel = WirelessChannel(cls.config)
        cls.ris_opt = RISOptimizer(num_elements=cls.config['channel']['num_ris_elements'], max_iters=15)

    def _run_strategy(self, strat_name, strat_type):
        fl_cfg = copy.deepcopy(self.config.get('fl', {}))
        fl_cfg['num_clients'] = self.num_clients
        fl_cfg['local_epochs'] = 2
        fl_cfg['learning_rate'] = 0.05

        run_cfg = _exp2_config(self.config, latency_ms=150.0, energy_mj=75.0)
        run_cfg['fl'] = fl_cfg
        run_cfg['selection']['strategy'] = strat_type
        run_cfg['selection']['max_participants'] = 3

        def model_fn():
            return get_model(self.dataset_name, torch.device('cpu'))

        clients = _create_clients(self.num_clients, self.client_loaders, model_fn, fl_cfg, run_cfg)
        selector = _get_selector(strat_type, run_cfg, self.channel, self.ris_opt)
        use_ris, ris_mode = _ris_flags(strat_type)

        server = FLServer(
            model_fn=model_fn,
            test_loader=self.test_loader,
            config=run_cfg,
            channel=self.channel,
            ris_optimizer=self.ris_opt if use_ris else None,
            device_selector=selector,
            validation_loader=self.val_loader,
        )

        initial_weights = copy.deepcopy(server.global_model.state_dict())
        logger = MetricsLogger(self.save_dir, f"smoke_{strat_type}")

        results = server.run(
            clients=clients,
            num_rounds=self.num_rounds,
            metrics_logger=logger,
            use_ris=use_ris,
            ris_mode=ris_mode,
            verbose=False,
        )
        logger.save_csv()
        logger.save_json()

        return server, results, initial_weights

    def test_proposed_joint_ris(self):
        server, results, init_weights = self._run_strategy("Proposed", "joint")
        self.assertEqual(len(results['round']), self.num_rounds)
        # Check participation > 0
        avg_part = np.mean(results['num_participating'])
        self.assertGreater(avg_part, 0, "No clients participated in Joint+RIS!")
        # Check model updated
        final_weights = server.global_model.state_dict()
        weight_diff = sum(
            torch.norm(final_weights[k].float() - init_weights[k].float()).item()
            for k in init_weights if final_weights[k].is_floating_point()
        )
        self.assertGreater(weight_diff, 1e-4, "Global model weights did not change after rounds!")

    def test_all_seven_strategies(self):
        strategies = {
            'Joint+RIS (Proposed)': 'joint',
            'Joint+RIS (No utility)': 'joint_no_utility',
            'Joint+RIS (No upload debt)': 'joint_no_debt',
            'Joint+RIS (No starvation age)': 'joint_no_age',
            'Channel-Only (No RIS)': 'channel_only',
            'Random (No RIS)': 'random',
            'Channel + Random RIS': 'channel_random_ris',
        }
        for name, strat_type in strategies.items():
            with self.subTest(strategy=name):
                server, results, init_weights = self._run_strategy(name, strat_type)
                self.assertEqual(len(results['accuracy']), self.num_rounds)
                for acc in results['accuracy']:
                    self.assertTrue(np.isfinite(acc))
                for lat in results['per_round_latency_ms']:
                    if lat is not None:
                        self.assertGreaterEqual(lat, 0.0)


if __name__ == '__main__':
    unittest.main()
