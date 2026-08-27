"""
RIS phase shift optimisation algorithms.

Implements element-wise Alternating Optimisation (AO) for:
  - Sum-SNR maximisation
  - Max-min fairness (min-SNR maximisation)
  - AirComp magnitude alignment

Core algorithm:
    For each RIS element n (fixing all others):
        theta_n* = angle( sum_k w_k * conj(a_{k,n}) * b_{k,n} )
    where:
        a_{k,n} = h_BR[:, n] * h_RU_k[n]   (contribution of element n)
        b_{k,n} = h_eff_k - a_{k,n} * phi[n]  (everything except element n)
"""

import numpy as np
from typing import Optional


class RISOptimizer:
    """Optimise RIS phase shifts for various objectives."""

    def __init__(self, num_elements: int, max_iters: int = 50,
                 tol: float = 1e-6):
        """
        Args:
            num_elements: Number of RIS elements (N).
            max_iters: Max AO iterations.
            tol: Convergence tolerance on ||phi_new - phi_old||.
        """
        self.N = int(num_elements)
        self.max_iters = int(max_iters)
        self.tol = float(tol)

    # ------------------------------------------------------------------ #
    #  Public methods                                                      #
    # ------------------------------------------------------------------ #
    def optimize(self, h_br: np.ndarray,
                 h_ru_list: list[np.ndarray],
                 h_d_list: list[np.ndarray],
                 mode: str = 'sum_snr') -> np.ndarray:
        """Generic entry point.

        Args:
            h_br: BS-RIS channel (M x N)
            h_ru_list: list of RIS-User channels, each (N x 1)
            h_d_list: list of direct channels, each (M x 1)
            mode: 'sum_snr', 'min_snr', or 'aircomp'

        Returns:
            phi: optimal phase vector (N,) with |phi_n| = 1
        """
        return self._element_wise_ao(h_br, h_ru_list, h_d_list, mode=mode)

    def optimize_sum_snr(self, h_br, h_ru_list, h_d_list) -> np.ndarray:
        """Maximise sum-SNR across users."""
        return self._element_wise_ao(h_br, h_ru_list, h_d_list, mode='sum_snr')

    def optimize_min_snr(self, h_br, h_ru_list, h_d_list) -> np.ndarray:
        """Max-min fairness: maximise minimum user SNR."""
        return self._element_wise_ao(h_br, h_ru_list, h_d_list, mode='min_snr')

    def optimize_for_aircomp(self, h_br, h_ru_list, h_d_list) -> np.ndarray:
        """Optimise for AirComp magnitude alignment."""
        return self._element_wise_ao(h_br, h_ru_list, h_d_list, mode='aircomp')

    def random_phases(self, n_elements: Optional[int] = None) -> np.ndarray:
        """Baseline: uniform random phase shifts, |phi_n| = 1."""
        n = n_elements if n_elements is not None else self.N
        theta = np.random.uniform(0, 2 * np.pi, n)
        return np.exp(1j * theta)

    # ------------------------------------------------------------------ #
    #  Core: element-wise AO                                               #
    # ------------------------------------------------------------------ #
    def _element_wise_ao(self, h_br: np.ndarray,
                         h_ru_list: list[np.ndarray],
                         h_d_list: list[np.ndarray],
                         mode: str = 'sum_snr') -> np.ndarray:
        """Element-wise alternating optimisation.

        For each element n, fix all other phases and find the optimal theta_n
        that maximises the chosen objective.

        Precomputed quantities per user k:
            A_k[:, n] = h_br[:, n] * h_ru_k[n]   — (M,) vector
            This lets us write h_eff_k = A_k @ phi + h_d_k
        """
        N = self.N
        K = len(h_ru_list)

        # Initialise with random phases
        phi = self.random_phases(N)

        # Precompute A_k (M x N) for each user
        # A_k[:, n] = h_br[:, n] * h_ru_k[n]
        A = []
        for k in range(K):
            h_ru_flat = h_ru_list[k].flatten()  # (N,)
            A.append(h_br * h_ru_flat[np.newaxis, :])  # broadcast (M, N)

        h_d_flat = [h_d_list[k].flatten() for k in range(K)]  # each (M,)

        for iteration in range(self.max_iters):
            phi_old = phi.copy()

            for n in range(N):
                if mode == 'sum_snr':
                    # Maximise sum_k ||h_eff_k||^2
                    # Optimal: phi[n] = angle( sum_k conj(a_{k,n})^T @ b_{k,n} )
                    val = 0.0 + 0.0j
                    for k in range(K):
                        a_kn = A[k][:, n]  # (M,)
                        # h_eff_k = A_k @ phi + h_d_k
                        h_eff_k = A[k] @ phi + h_d_flat[k]
                        b_kn = h_eff_k - a_kn * phi[n]  # remove element n
                        val += np.vdot(a_kn, b_kn)  # conj(a_kn)^T @ b_kn
                    if np.abs(val) > 1e-15:
                        phi[n] = val / np.abs(val)

                elif mode == 'min_snr':
                    # Find the user with minimum ||h_eff_k|| and optimise for them
                    min_gain = np.inf
                    min_k = 0
                    a_kn_all = []
                    b_kn_all = []
                    for k in range(K):
                        a_kn = A[k][:, n]
                        h_eff_k = A[k] @ phi + h_d_flat[k]
                        b_kn = h_eff_k - a_kn * phi[n]
                        gain = np.linalg.norm(h_eff_k) ** 2
                        if gain < min_gain:
                            min_gain = gain
                            min_k = k
                        a_kn_all.append(a_kn)
                        b_kn_all.append(b_kn)

                    # Optimise for the worst-case user
                    val = np.vdot(a_kn_all[min_k], b_kn_all[min_k])
                    if np.abs(val) > 1e-15:
                        phi[n] = val / np.abs(val)

                elif mode == 'aircomp':
                    # Objective: minimise channel-gain spread (alignment)
                    # Heuristic: weighted sum-SNR where weights inversely
                    # proportional to current channel gain
                    gains = []
                    a_kn_all = []
                    b_kn_all = []
                    for k in range(K):
                        a_kn = A[k][:, n]
                        h_eff_k = A[k] @ phi + h_d_flat[k]
                        b_kn = h_eff_k - a_kn * phi[n]
                        gains.append(np.linalg.norm(h_eff_k) ** 2 + 1e-15)
                        a_kn_all.append(a_kn)
                        b_kn_all.append(b_kn)

                    # Weight inversely by gain → boost weak users
                    val = 0.0 + 0.0j
                    for k in range(K):
                        w_k = 1.0 / gains[k]
                        val += w_k * np.vdot(a_kn_all[k], b_kn_all[k])
                    if np.abs(val) > 1e-15:
                        phi[n] = val / np.abs(val)

            # Convergence check
            diff = float(np.linalg.norm(phi - phi_old).real)
            if diff < self.tol:
                break

        return phi
