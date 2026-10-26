"""
Wireless channel models for RIS-assisted communications.

System model:
    BS (M antennas) --- direct link ---> User k
    BS (M antennas) --- h_BR ---> RIS (N elements) --- h_RU_k ---> User k

    Effective channel for user k:
        h_eff_k = H_BR^H @ Phi @ h_RU_k + h_d_k

    where Phi = diag(exp(j * theta_1), ..., exp(j * theta_N))
"""

import numpy as np
from typing import Optional


class WirelessChannel:
    """Rician/Rayleigh fading channel model for RIS-assisted communications."""

    def __init__(self, config: dict):
        """Initialize the channel model.

        Args:
            config: Full config dict (with 'channel' key) OR a flat channel config dict.
        """
        # Support both nested and flat config
        ch = config.get('channel', config)

        self.M = ch.get('num_antennas_bs', 4)           # BS antennas
        self.N = ch.get('num_ris_elements', 64)          # RIS elements
        self.model_type = ch.get('model', 'rician')      # rician or rayleigh

        # Rician K-factor (dB → linear)
        self.K_dB = ch.get('rician_k_factor', 10.0)
        self.K = 10 ** (self.K_dB / 10.0)

        # Carrier frequency
        self.fc_ghz = ch.get('carrier_freq_ghz', 28.0)
        self.wavelength = 0.3 / self.fc_ghz  # c / f in metres

        # Bandwidth
        self.bandwidth_hz = ch.get('bandwidth_mhz', 10.0) * 1e6

        # Noise spectral density N0 (dBm/Hz) and total noise power (Watts) = N0 * B
        # noise_power_dbm in config represents the thermal noise spectral density N0 = -174 dBm/Hz
        noise_psd_dbm = float(ch.get('noise_spectral_density_dbm_per_hz',
                                     ch.get('noise_power_dbm', -174.0)))
        # Total noise power in Watts = N0 * B = 10^((N0_dBm - 30)/10) * B_Hz
        self.noise_power_w = 10 ** ((noise_psd_dbm - 30) / 10.0) * self.bandwidth_hz

        # Path loss exponents
        self.alpha_direct = ch.get('path_loss_exponent_direct', 3.5)
        self.alpha_br = ch.get('path_loss_exponent_br', 2.2)
        self.alpha_ru = ch.get('path_loss_exponent_ru', 2.5)

        # Reference path loss at d_0 = 1 m (free space at carrier frequency)
        self.d_0 = 1.0
        self.PL_0_dB = 20 * np.log10(4 * np.pi * self.d_0 / self.wavelength)

        # Distances (metres)
        self.d_bs_ris = ch.get('d_bs_ris', 50.0)
        self.d_ris_user_min = ch.get('d_ris_user_min', 10.0)
        self.d_ris_user_max = ch.get('d_ris_user_max', 100.0)
        self.d_bs_user_min = ch.get('d_bs_user_min', 50.0)
        self.d_bs_user_max = ch.get('d_bs_user_max', 200.0)

        # Direct-link blockage (mmWave / NLoS): extra attenuation so RIS can matter
        self.direct_blockage_prob = float(ch.get('direct_blockage_prob', 0.0))
        self.direct_blockage_db = float(ch.get('direct_blockage_attenuation_db', 30.0))

    # ------------------------------------------------------------------ #
    #  Path loss                                                          #
    # ------------------------------------------------------------------ #
    def _path_loss_linear(self, distance: float, exponent: float) -> float:
        """Compute path loss in linear scale.

        PL(d) = PL_0 + 10 * alpha * log10(d / d_0)   [dB]
        Returns PL in linear (< 1).
        """
        pl_dB = self.PL_0_dB + 10 * exponent * np.log10(np.maximum(distance, self.d_0) / self.d_0)
        return 10 ** (-pl_dB / 10.0)

    # ------------------------------------------------------------------ #
    #  Steering vectors (ULA)                                             #
    # ------------------------------------------------------------------ #
    def _ula_steering(self, num_elements: int, angle_rad: float) -> np.ndarray:
        """ULA steering vector: a(theta) = [1, e^{j*pi*sin(theta)}, ...].

        Half-wavelength spacing assumed (d_ant = lambda/2).
        Returns shape (num_elements, 1).
        """
        n = np.arange(num_elements).reshape(-1, 1)
        return np.exp(1j * np.pi * n * np.sin(angle_rad))

    # ------------------------------------------------------------------ #
    #  Channel generation                                                 #
    # ------------------------------------------------------------------ #
    def generate_bs_ris_channel(self) -> np.ndarray:
        """Generate BS-to-RIS channel H_BR (M x N).

        Rician: H = sqrt(PL) * [sqrt(K/(K+1)) * H_LoS + sqrt(1/(K+1)) * H_NLoS]
        LoS component uses ULA steering vectors.
        """
        pl = self._path_loss_linear(self.d_bs_ris, self.alpha_br)

        if self.model_type == 'rayleigh':
            H_fading = (np.random.randn(self.M, self.N)
                        + 1j * np.random.randn(self.M, self.N)) / np.sqrt(2)
        else:  # rician
            # Random AoD/AoA for LoS component
            aod = np.random.uniform(-np.pi / 2, np.pi / 2)
            aoa = np.random.uniform(-np.pi / 2, np.pi / 2)
            a_bs = self._ula_steering(self.M, aod)    # (M, 1)
            a_ris = self._ula_steering(self.N, aoa)   # (N, 1)
            H_LoS = a_bs @ a_ris.conj().T             # (M, N)

            H_NLoS = (np.random.randn(self.M, self.N)
                      + 1j * np.random.randn(self.M, self.N)) / np.sqrt(2)

            H_fading = (np.sqrt(self.K / (self.K + 1)) * H_LoS
                        + np.sqrt(1 / (self.K + 1)) * H_NLoS)

        return np.sqrt(pl) * H_fading

    def generate_ris_user_channels(self, num_users: int,
                                    user_distances: Optional[np.ndarray] = None
                                    ) -> list[np.ndarray]:
        """Generate RIS-to-User channels h_RU_k (N x 1) for each user.

        Rayleigh fading (NLoS dominated for user links).
        """
        if user_distances is None:
            user_distances = np.random.uniform(
                self.d_ris_user_min, self.d_ris_user_max, num_users)

        channels = []
        for d in user_distances:
            pl = self._path_loss_linear(d, self.alpha_ru)
            h = (np.random.randn(self.N, 1)
                 + 1j * np.random.randn(self.N, 1)) / np.sqrt(2)
            channels.append(np.sqrt(pl) * h)
        return channels

    def generate_direct_channels(self, num_users: int,
                                  user_distances: Optional[np.ndarray] = None
                                  ) -> list[np.ndarray]:
        """Generate direct BS-to-User channels h_d_k (M x 1) for each user.

        Rayleigh fading (NLoS, heavily attenuated).
        """
        if user_distances is None:
            user_distances = np.random.uniform(
                self.d_bs_user_min, self.d_bs_user_max, num_users)

        channels = []
        for d in user_distances:
            pl = self._path_loss_linear(d, self.alpha_direct)
            if np.random.rand() < self.direct_blockage_prob:
                pl *= 10 ** (-self.direct_blockage_db / 10.0)
            h = (np.random.randn(self.M, 1)
                 + 1j * np.random.randn(self.M, 1)) / np.sqrt(2)
            channels.append(np.sqrt(pl) * h)
        return channels

    # ------------------------------------------------------------------ #
    #  Effective channel                                                   #
    # ------------------------------------------------------------------ #
    def compute_effective_channel(self, h_br: np.ndarray,
                                   h_ru_k: np.ndarray,
                                   h_d_k: np.ndarray,
                                   phi: np.ndarray) -> np.ndarray:
        """Compute effective channel for user k.

        h_eff_k = H_BR @ diag(phi) @ h_RU_k + h_d_k

        Args:
            h_br:  BS-RIS channel (M x N)
            h_ru_k: RIS-User channel (N x 1)
            h_d_k: Direct channel (M x 1)
            phi:   Phase shift vector (N,) with |phi_n| = 1

        Returns:
            h_eff: (M x 1) complex vector
        """
        Phi = np.diag(phi)                       # (N x N)
        h_eff = h_br @ Phi @ h_ru_k + h_d_k     # (M x 1)
        return h_eff

    # ------------------------------------------------------------------ #
    #  SNR, rate, outage                                                   #
    # ------------------------------------------------------------------ #
    def compute_snr(self, h_eff: np.ndarray,
                    tx_power_dbm: float = 23.0) -> float:
        """Compute received SNR (linear).

        SNR = p_k * ||h_eff||^2 / sigma^2
        """
        tx_power_w = 10 ** ((tx_power_dbm - 30) / 10.0)
        channel_gain = np.linalg.norm(h_eff) ** 2
        return tx_power_w * channel_gain / self.noise_power_w

    def snr_to_db(self, snr_linear: float) -> float:
        """Convert linear SNR to dB."""
        return 10 * np.log10(max(snr_linear, 1e-20))

    def compute_achievable_rate(self, snr_linear: float) -> float:
        """Shannon capacity R = B * log2(1 + SNR) [bps]."""
        return self.bandwidth_hz * np.log2(1 + snr_linear)

    def compute_outage_probability(self, snr_samples: list[float],
                                    threshold_dB: float) -> float:
        """P(SNR < threshold)."""
        threshold_lin = 10 ** (threshold_dB / 10.0)
        arr = np.array(snr_samples)
        return float(np.mean(arr < threshold_lin))

    # ------------------------------------------------------------------ #
    #  Convenience: generate all channels for one FL round                 #
    # ------------------------------------------------------------------ #
    def generate_all_channels(self, num_users: int) -> dict:
        """Generate complete channel realisation for one communication round.

        Returns:
            dict with keys: h_br, h_ru, h_d, dist_ris_user, dist_bs_user
        """
        dist_ris_user = np.random.uniform(
            self.d_ris_user_min, self.d_ris_user_max, num_users)
        dist_bs_user = np.random.uniform(
            self.d_bs_user_min, self.d_bs_user_max, num_users)

        h_br = self.generate_bs_ris_channel()
        h_ru = self.generate_ris_user_channels(num_users, dist_ris_user)
        h_d = self.generate_direct_channels(num_users, dist_bs_user)

        return {
            'h_br': h_br,
            'h_ru': h_ru,
            'h_d': h_d,
            'dist_ris_user': dist_ris_user,
            'dist_bs_user': dist_bs_user,
        }
