"""Federated Learning client."""

import torch
import torch.nn as nn
import torch.optim as optim

class FLClient:
    """Federated Learning client with wireless transmission simulation."""
    def __init__(self, client_id, dataloader, model_fn, config):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model_fn().to(self.device)
        self.config = config
        self.lr = config.get("lr", 0.01)
        self.momentum = config.get("momentum", 0.5)
        self.epochs = config.get("local_epochs", 1)
        
    def train(self, global_params: dict) -> tuple[dict, int, dict]:
        """Local training.
        1. Load global parameters
        2. Train for E local epochs on local data
        3. Compute gradient update (delta = local - global)
        4. Return (parameters_dict, num_samples, metrics_dict)
        """
        self.model.load_state_dict(global_params)
        self.model.train()
        
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        criterion = nn.CrossEntropyLoss()
        
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            for batch_idx, (data, target) in enumerate(self.dataloader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss) if batch_loss else 0.0)

        num_samples = len(self.dataloader.dataset)
        metrics = {"loss": (sum(epoch_loss) / len(epoch_loss)) if epoch_loss else 0.0}
        
        # In typical FedAvg, return the updated weights. 
        # If delta is strictly needed, calculate delta = local - global. 
        # We will return the updated state_dict directly for simplicity, which works for FedAvg.
        return self.model.state_dict(), num_samples, metrics
        
    def compute_transmission_latency(self, model_size_bits: int, rate_bps: float) -> float:
        """T_comm = model_size / rate."""
        if rate_bps <= 0:
            return float('inf')
        return float(model_size_bits) / rate_bps
        
    def compute_computation_latency(self, num_samples: int, cycles_per_sample: float, cpu_freq_hz: float) -> float:
        """T_comp = num_samples * cycles_per_sample / cpu_freq."""
        return (float(num_samples) * cycles_per_sample) / cpu_freq_hz
        
    def compute_energy(self, tx_power_w: float, tx_time_s: float, comp_energy_per_sample: float, num_samples: int) -> float:
        """E = P_tx * T_tx + E_comp_per_sample * num_samples."""
        return (tx_power_w * tx_time_s) + (comp_energy_per_sample * float(num_samples))
