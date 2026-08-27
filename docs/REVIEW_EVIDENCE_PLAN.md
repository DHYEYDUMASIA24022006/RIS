# RIS-Assisted Wireless Federated Learning: Review Evidence Plan

This document maps the project evidence to the two assessment rubrics. A short
`quick` run is an environment demonstration only; it must not be presented as
the final quantitative result.

## Review 1: Problem Formulation and Design (40 marks)

### Project setting and motivation

The system is an uplink wireless federated-learning network with one multi-antenna
base station/parameter server, a passive reconfigurable intelligent surface (RIS),
and single-antenna edge clients. Clients train locally and upload model updates over
fading links. A blocked direct path causes unreliable uploads; the RIS shapes the
cascaded BS-RIS-client path to improve signal quality. The work supports SDG 9
(resilient digital infrastructure) and SDG 4 (privacy-preserving distributed AI
for resource-constrained learning settings).

### One-sentence problem statement

How can RIS configuration and wireless-aware client participation reduce dropped
federated-learning updates and improve learning efficiency under fading, latency,
energy, and non-IID-data constraints?

### Measurable objectives

1. Quantify the effect of optimized RIS phases on SNR, outage, successful uploads,
   FL accuracy, and rounds-to-target accuracy against no-RIS and random-RIS baselines.
2. Design and evaluate joint client selection and RIS optimization subject to
   per-round latency and per-device energy constraints.
3. Evaluate RIS-assisted over-the-air aggregation (AirComp) and quantify its
   aggregation MSE and learning accuracy under IID and Dirichlet non-IID partitions.
4. Report the fairness of actual successful client uploads using Jain's fairness index.

### Scope boundaries

The project is a reproducible Python simulation, not an RF hardware prototype.
It models narrowband Rician/Rayleigh links and does not claim a 3GPP-compliant
channel model, hardware RIS imperfections, adversarial clients, or privacy proofs.

### Methods and named baselines

* Experiment 1: ideal channel, fading/no RIS, fading/random RIS, and
  fading/optimized RIS.
* Experiment 2: proposed joint selection + optimized RIS, channel-only/no RIS,
  random/no RIS, and channel-only with random RIS. All strategies use the same
  maximum participant count.
* Experiment 3: FedAvg ideal, FedAvg + RIS, FedAvg/no RIS, AirComp + RIS, and
  AirComp/no RIS. AirComp uses sample-weighted model deltas, channel inversion,
  a transmit-power cap, and receiver AWGN.

### Required Review-1 evidence before presentation

* Literature comparison table with at least 25 verified recent journal papers.
  Include year, venue, scenario, data/testbed, optimization variables, metrics,
  and a limitation. At least 18 should be from 2021 onward.
* A thematic synthesis, not 25 sequential summaries: RIS-WFL, energy/latency-aware
  scheduling, AirComp, and non-IID/fairness.
* Research gaps traced to sources: (G1) RIS-assisted scheduling is often studied
  without matched budgeted baselines; (G2) AirComp reliability and non-IID learning
  are often reported separately; (G3) successful-upload fairness is usually not
  jointly reported with learning and physical-layer metrics.
* System architecture and three algorithm-flow diagrams; include the physical
  channel, FL server/clients, RIS optimizer, scheduler, AirComp branch, logger,
  and results layer.

## Review 2: Implementation and Dissemination (60 marks)

### Environment to show live

* Python, PyTorch, NumPy, Matplotlib, Seaborn, PyYAML, and the dependency versions
  in `requirements.txt`.
* A CPU smoke run: `python main.py --experiment quick`.
* The central scenario configuration: `config/config.yaml`.
* The actual output directory: `results/exp1`, `results/exp2`, and `results/exp3`.

### Required final experimental protocol

For every final figure, use at least three independent seeds and report mean and
standard deviation (or 95% confidence intervals). Keep dataset, client count,
rounds, model, partition alpha, SNR threshold, RIS size, and budgets fixed across
the compared methods. Do not compare runs with unequal numbers of allowed clients.

### Metrics, units, and measurement method

* Test accuracy (%), loss (cross-entropy), and rounds to target accuracy (rounds).
* Aggregation MSE (normalized squared error versus exact sample-weighted FedAvg).
* Mean/min/max SNR (dB), outage probability (fraction), and mean uplink rate (bps).
* Per-round latency (ms) and mean per-client energy (mJ), including communication
  and configured computation energy.
* Scheduled and successful clients (count), plus Jain successful-upload fairness
  (0 to 1).

### Dataset specification

MNIST and CIFAR-10 are loaded through TorchVision. MNIST contains 60,000 training
and 10,000 test grayscale 28x28 images across 10 balanced classes; CIFAR-10 contains
50,000 training and 10,000 test RGB 32x32 images across 10 balanced classes. Both
use the standard TorchVision dataset terms. MNIST is used for smoke tests; CIFAR-10
is the primary final experiment. Normalisation is defined in `fl/data_partition.py`.
IID allocation is random and equal-sized; non-IID allocation uses class-wise
Dirichlet splitting with the reported alpha value. The standard test set is held out
from all local client training.

### Result-analysis requirements

Every final plot must have a caption, a named baseline, units, legend, seed count,
and interpretation. Discuss: (1) outage-dominated low-SNR behavior, (2) feasibility
loss under tight energy/latency budgets, (3) AirComp distortion versus communication
saving, (4) non-IID degradation, and (5) fairness trade-offs. State limitations
honestly; the quick run is not statistically meaningful.

### Dissemination deliverables

1. Review-1 slide deck and report section with verified 25-paper survey.
2. Review-2 deck containing reproducible setup, final figures/tables, limitations,
   and conclusion quantified from the final seed-averaged results.
3. A full manuscript in the target venue template. Claim only "draft complete"
   until a submission ID or acceptance evidence exists.
