"""Baseline device selection strategies for comparison."""

import numpy as np


class RandomSelector:
    """Randomly select K clients each round (no channel info used)."""

    def __init__(self, k: int):
        self.k = k

    def select(self, snr_list, k=None) -> list[int]:
        """Randomly select k clients.

        Args:
            snr_list: Per-client SNR (ignored for random selection).
            k: Number to select (overrides self.k if given).
        """
        num_clients = len(snr_list) if hasattr(snr_list, '__len__') else snr_list
        k = k if k is not None else self.k
        k = min(k, num_clients)
        return list(np.random.choice(num_clients, k, replace=False))


class ChannelOnlySelector:
    """Select top-K clients by SNR (channel-aware, no RIS optimisation)."""

    def __init__(self, k: int):
        self.k = k

    def select(self, snr_list, k=None) -> list[int]:
        """Select k clients with highest SNR.

        Args:
            snr_list: Per-client SNR values (linear).
            k: Number to select (overrides self.k if given).
        """
        snr_arr = np.array(snr_list)
        k = k if k is not None else self.k
        k = min(k, len(snr_arr))
        return np.argsort(snr_arr)[-k:].tolist()


class ChannelWithFixedRISSelector:
    """Select top-K clients by SNR computed with fixed random RIS phases."""

    def __init__(self, k: int):
        self.k = k

    def select(self, snr_list, k=None) -> list[int]:
        """Same as ChannelOnlySelector (RIS phases set externally)."""
        snr_arr = np.array(snr_list)
        k = k if k is not None else self.k
        k = min(k, len(snr_arr))
        return np.argsort(snr_arr)[-k:].tolist()
