"""Over-the-Air Computation for Federated Learning."""

import numpy as np

class AirCompAggregator:
    """Over-the-Air Computation aggregator with RIS-assisted magnitude alignment.
    
    In AirComp, all selected clients transmit simultaneously.
    The server receives: y = sum_k h_eff_k * sqrt(p_k) * s_k + noise
    where s_k is the normalised gradient of client k.
    
    With ideal channel inversion: p_k = eta / |h_eff_k|^2
    RIS helps align |h_eff_k| across clients to reduce power variance.
    """
    
    def __init__(self, config):
        self.config = config
        
    def aggregate(self, client_gradients: list[np.ndarray], channel_gains: list[complex],
                  noise_power: float, max_power: float,
                  weights: list[float] | None = None) -> np.ndarray:
        """Simulate AirComp aggregation.
        The transmit precoder is b_k = sqrt(eta) / h_k, where
        eta = P_max min_k |h_k|^2.  Therefore |b_k|^2 <= P_max and every
        non-zero effective channel contributes exactly sqrt(eta) times its
        update at the receiver.  The receiver divides by sqrt(eta) and K.

        ``weights`` are FedAvg sample weights.  Client k transmits K*w_k
        times its update, so the recovered value is sum_k w_k * update_k.
        Returns aggregated gradient vector.
        """
        K = len(client_gradients)
        if K == 0:
            return None
            
        dim = client_gradients[0].shape[0]
        if len(channel_gains) != K:
            raise ValueError("channel_gains must contain one gain per client")
        if weights is None:
            weights = [1.0 / K] * K
        if len(weights) != K or not np.isclose(np.sum(weights), 1.0):
            raise ValueError("weights must be one normalised value per client")

        gains_sq = np.abs(np.asarray(channel_gains, dtype=complex)) ** 2
        positive_gains = gains_sq[gains_sq > 1e-20]
        if positive_gains.size == 0 or max_power <= 0:
            return np.zeros(dim, dtype=float)

        eta = float(max_power * np.min(positive_gains))
        received_signal = np.zeros(dim, dtype=complex)
        
        for k in range(K):
            s_k = client_gradients[k]
            h_eff_k = channel_gains[k]
            # Phase and magnitude inversion under the individual power cap.
            if np.abs(h_eff_k) > 1e-12:
                precoder = np.sqrt(eta) / h_eff_k
            else:
                precoder = 0.0
            received_signal += (K * weights[k]) * s_k * precoder * h_eff_k
            
        # Add AWGN noise
        noise_std = np.sqrt(noise_power / 2.0)
        noise = np.random.normal(0, noise_std, dim) + 1j * np.random.normal(0, noise_std, dim)
        received_signal += noise
        
        # Recover the weighted average. Inputs are real-valued model updates.
        estimated_avg = np.real(received_signal) / np.sqrt(eta) / K
        return estimated_avg
        
    def compute_aggregation_mse(self, true_avg: np.ndarray, estimated_avg: np.ndarray) -> float:
        """MSE = ||g_hat - g_true||^2 / ||g_true||^2 (normalised)."""
        num = np.linalg.norm(estimated_avg - true_avg)**2
        den = np.linalg.norm(true_avg)**2 + 1e-12
        return float(num / den)
        
    def compute_power_allocation(self, channel_gains: list[complex], max_power: float) -> list[float]:
        """Channel inversion power control.
        p_k = min(eta / |h_k|^2, P_max)
        eta chosen so weakest user hits P_max.
        """
        K = len(channel_gains)
        channel_mags_sq = np.abs(channel_gains)**2
        
        # To make the weakest user hit P_max exactly:
        # min(|h_k|^2) * P_max = eta
        min_channel_mag_sq = np.min(channel_mags_sq)
        eta = max_power * min_channel_mag_sq
        
        tx_powers = []
        for k in range(K):
            if channel_mags_sq[k] > 1e-12:
                p_k = min(eta / channel_mags_sq[k], max_power)
            else:
                p_k = max_power
            tx_powers.append(float(p_k))
            
        return tx_powers
