# RIS-Assisted Wireless Federated Learning

Welcome to the official repository for the **RIS-Assisted Wireless Federated Learning** project. This project provides a comprehensive, end-to-end simulation framework for evaluating the integration of Reconfigurable Intelligent Surfaces (RIS) in wireless Federated Learning (FL) systems.

The core motivation is to overcome the communication bottleneck and mitigate the harsh propagation environments common in high-frequency (e.g., mmWave) communications by intelligently reconfiguring the wireless environment using RIS, while simultaneously optimizing client participation to meet strict energy and latency constraints.

---

## 1. System Model and Architecture

The physical communication layer consists of a Base Station (BS) equipped with $M$ antennas, a Reconfigurable Intelligent Surface (RIS) equipped with $N$ passive reflecting elements, and $K$ single-antenna edge users (clients) participating in the federated training process.

```text
       +-------------------+
       |                   |
       |   Base Station    | (M antennas)
       |   (FL Server)     |
       |                   |
       +---------+---------+
                 |
                 | h_d (Direct Link) - often blocked or highly attenuated
                 |
                 v
       +-------------------+        h_RU        +-------------------+
       |                   | <----------------- |                   |
       |   Edge Users      |                    |       RIS         | (N elements)
       |   (K clients)     | -----------------> |                   |
       +-------------------+                    +-------------------+
                 ^                                        |
                 |                                        |
                 +----------------------------------------+
                             h_BR (BS-RIS Link)
```

In typical mmWave scenarios, the direct link ($h_d$) might suffer from severe blockage. The RIS provides a controlled virtual line-of-sight (LoS) path, significantly enhancing the signal-to-noise ratio (SNR) for the selected clients.

---

## 2. Mathematical Formulation

### 2.1 Channel Model
The effective channel $h_{eff,k} \in \mathbb{C}^{M \times 1}$ for user $k$ is given by the superposition of the direct link and the cascaded RIS link:
$$ h_{eff,k} = \mathbf{H}_{BR}^H \mathbf{\Phi} \mathbf{h}_{RU,k} + \mathbf{h}_{d,k} $$
Where:
- $\mathbf{H}_{BR} \in \mathbb{C}^{N \times M}$: Channel matrix from the BS to the RIS.
- $\mathbf{h}_{RU,k} \in \mathbb{C}^{N \times 1}$: Channel vector from the RIS to user $k$.
- $\mathbf{h}_{d,k} \in \mathbb{C}^{M \times 1}$: Direct channel vector from the BS to user $k$.
- $\mathbf{\Phi} = \text{diag}(e^{j\theta_1}, \dots, e^{j\theta_N})$: RIS phase shift matrix, where $\theta_n \in [0, 2\pi)$ is the phase shift induced by the $n$-th element.

### 2.2 Optimization Problem
The overall objective is to minimize the global FL loss function $F(\mathbf{w})$ while ensuring reliable communication and satisfying device-level constraints:
$$ \min_{\mathbf{w}, \mathbf{\Phi}, \mathcal{S}} F(\mathbf{w}) = \frac{1}{|\mathcal{S}|} \sum_{k \in \mathcal{S}} F_k(\mathbf{w}) $$
Subject to:
1. **Latency Constraints**: Total time (computation + communication) per round $\le T_{max}$.
2. **Energy Constraints**: Total energy (computation + communication) per device $\le E_{max}$.
3. **RIS Phase Constraints**: $|\Phi_{n,n}| = 1, \forall n = 1, \dots, N$.

This forms a mixed-integer non-linear programming (MINLP) problem, which we decouple and solve using alternating optimization (AO) techniques and greedy device selection heuristics.

---

## 3. Installation and Setup

### 3.1 Prerequisites
- Python 3.10+
- CUDA-enabled GPU (Highly recommended for deep learning workloads)
- Git

### 3.2 Installation Steps
1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/ris-assisted-wireless-fl.git
   cd ris-assisted-wireless-fl
   ```

2. **Create a Python virtual environment:**
   ```bash
   python -m venv venv
   # On Linux/macOS
   source venv/bin/activate
   # On Windows
   venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 4. Configuration

All system parameters, channel models, learning hyperparameters, and constraints are centrally managed via the `config/config.yaml` file. Key sections include:
- `system`: Random seeds, device placement (CPU/GPU), output directories.
- `channel`: Antenna counts, Rician K-factors, path loss exponents.
- `ris`: Optimization algorithms, phase quantization.
- `fl`: Rounds, clients, learning rates, target accuracies.
- `data`: Dataset selection, non-IID partitioning (Dirichlet).
- `selection`: Budgets for latency and energy.

Modify this file to customize your experiment parameters without changing the underlying code.

---

## 5. Running Experiments

The framework provides specialized scripts to launch different experimental scenarios. Ensure your virtual environment is activated before running.

### 5.1 Experiment 1: Digital FL with and without RIS
Evaluate the reliability and convergence benefit of an optimised RIS over a blocked, fading digital uplink. Set `data.partition` to `iid` or `noniid` and use `data.exp1_dirichlet_alpha` for the selected non-IID setting.
```bash
python main.py --experiment baseline
```
**Focus:** Accuracy progression, loss decay, communication rounds to reach target accuracy.

### 5.2 Experiment 2: Joint Device Selection and RIS Design
Compare the proposed joint design against random, channel-only, and random-RIS baselines under matched maximum participant counts and latency/energy budgets.
```bash
python main.py --experiment joint
```
**Focus:** Scheduled and successful clients, accuracy, latency, energy, outage, and Jain fairness of actual successful uploads.

### 5.3 Experiment 3: RIS-Assisted AirComp under Non-IID Data
Evaluate in-loop analog aggregation. AirComp transmits sample-weighted model deltas using channel inversion under a transmit-power cap; the server recovers the weighted aggregate after AWGN.
```bash
python main.py --experiment aircomp
```
**Focus:** AirComp MSE versus RIS size and SNR, AirComp versus FedAvg, IID versus Dirichlet non-IID accuracy, and successful-upload fairness.

For a short end-to-end smoke test on MNIST:
```bash
python main.py --experiment quick
```

---

## 6. Expected Results and Output

All experimental results are logged securely in the `results/` directory (created automatically). The output includes:
- **Metrics Logs:** Detailed `.csv` and `.json` files tracking accuracy, loss, successful uploads, aggregation MSE, rate, latency, energy, outage, and upload fairness.
- **Visualizations:** Publication-quality `.pdf` plots (customizable via `config.yaml`) suitable for direct inclusion in LaTeX documents.
- **Experiment summaries:** Aggregate JSON files for the budget sweeps and AirComp studies.

---

## 7. Project Structure

```text
RIS-Assisted-Wireless-Federated-Learning/
├── config/
│   ├── config.yaml          # Centralized configuration file
│   └── config_fast.yaml     # Fast smoke-test configuration
├── data/                    # Auto-downloaded datasets (MNIST, CIFAR-10)
├── channel/
│   ├── channel_model.py     # Wireless channel generation (Rician, Rayleigh)
│   └── ris_optimizer.py     # RIS phase shift optimization (AO)
├── fl/
│   ├── aircomp.py           # Over-the-air computation (AirComp)
│   ├── client.py            # FL client local training
│   ├── data_partition.py    # IID / Dirichlet non-IID partitioning
│   ├── models.py            # CNN architectures (MNIST, CIFAR-10)
│   └── server.py            # FL server with wireless aggregation
├── optimisation/
│   ├── joint_design.py      # Joint device selection + RIS optimization
│   └── baselines.py         # Baseline selection strategies
├── experiments/
│   ├── exp1_baseline.py     # Experiment 1: FL convergence ± RIS
│   ├── exp2_joint_design.py # Experiment 2: Joint client + RIS design
│   └── exp3_aircomp.py      # Experiment 3: AirComp evaluation
├── analysis/
│   └── compare_methods.py   # Cross-experiment comparison & plotting
├── metrics/
│   └── metrics_logger.py    # Standardized metric tracking and I/O
├── utils/
│   └── plotting.py          # Publication-quality plotting utilities
├── matlab/
│   └── channel_gen.m        # MATLAB channel generation script
├── docs/
│   ├── REVIEW_EVIDENCE_PLAN.md
│   └── RUNBOOK.md
├── results/                 # Generated experiment outputs (git-ignored)
├── main.py                  # Main entry point
├── run_full_evaluation.py   # Multi-seed evaluation runner
├── requirements.txt         # Dependency manifest
├── LICENSE                  # MIT License
└── README.md                # Repository documentation
```

---

## 8. Key References

This implementation builds upon the foundational research in both wireless communications and distributed machine learning:

1. **RIS Communications:** Wu, Q., & Zhang, R. (2019). Intelligent Reflecting Surface Enhanced Wireless Network via Joint Active and Passive Beamforming. *IEEE Transactions on Wireless Communications*.
2. **Energy-Efficient FL:** Yang, Z., et al. (2020). Energy Efficient Federated Learning Over Wireless Communication Networks. *IEEE Transactions on Wireless Communications*.
3. **Over-the-Air FL:** Zhu, G., et al. (2020). Broadband Analog Aggregation for Low-Latency Federated Edge Learning. *IEEE Transactions on Wireless Communications*.

---

## 9. Sustainable Development Goals (SDGs) Alignment

This research actively contributes to the United Nations Sustainable Development Goals:
- **Goal 9: Industry, Innovation, and Infrastructure:** By advancing state-of-the-art wireless communication architectures (RIS, 6G) that are both scalable and resilient.
- **Goal 4: Quality Education:** By supporting privacy-preserving distributed learning across resource-constrained participants.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
