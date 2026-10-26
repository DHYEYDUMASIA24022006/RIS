# Revised M2 Methodology

## Objective

M2 coordinates client participation and RIS phase design on each federated
round. The scheduler first enforces per-client latency and energy limits, then
ranks only feasible clients. It seeks useful updates while reducing repeated
exclusion, subject to a maximum participant count. The procedure is a practical
greedy heuristic for a mixed-integer, non-convex problem; it does not claim a
global optimum.

## Per-round model

For client (k), the effective channel under RIS phases \(\phi\) is modeled as
\(h_k^{eff}=h_{d,k}+H_{BR}^{H}\operatorname{diag}(e^{j\phi})h_{RU,k}\).
The simulator estimates achievable rate from this channel, then estimates
completion time as local computation time plus model-upload time. It estimates
energy as transmit energy plus configured per-sample computation energy.
Client (k) is eligible only when both estimates satisfy \(T_k\leq T_{max}\)
and \(E_k\leq E_{max}\). These quantities are analytical simulation estimates,
not measurements from physical devices.

## Learning and participation priority

The scheduler keeps three histories for each client:

1. **Validation utility:** an exponential moving average of validation-loss
   improvement after successful local training. The test set is not used for
   scheduling.
2. **Successful-upload debt:** clients with fewer successful uploads receive a
   higher rank.
3. **Starvation age:** the number of rounds since the last successful upload,
   capped by a configured maximum. The age increases for every client each
   round and resets only after a successful upload.

Each history is converted to a rank in [0, 1]. The priority is the normalized
weighted sum. Default weights are 0.65 utility, 0.20 upload debt, and 0.15 age;
all are configurable and normalized by the scheduler. A seeded random tie key
breaks exact ties so client IDs do not receive a persistent advantage.

## Alternating selection and phase design

1. Initialize RIS phases using the existing RIS optimizer and priority weights.
2. Estimate each client's rate, latency, and energy; remove clients that fail
   either hard budget.
3. Greedily select up to the participant cap among the feasible clients.
4. Re-optimize RIS phases for the selected clients using priority-weighted sum
   SNR, then repeat selection and phase optimization until selection stabilizes
   or the iteration limit is reached.
5. Recompute feasibility under the final phases. Remove violations and refill
   open slots from the remaining clients feasible under those same final phases.
   The returned set therefore meets modeled per-client limits.
6. Train the selected clients. Only successful uploads enter sample-weighted
   FedAvg. Evaluate candidate updates on the held-out validation set and update
   histories for the next round. If none are feasible, record an idle round.

## Implementation repairs and additions

- Replaced client-ID tie preference with repeatable seeded randomized ties.
- Added bounded rounds-since-success memory to surface clients repeatedly
  missed by the scheduler. Upload debt remains, so both long-term success count
  and recent starvation inform scheduling.
- Added validation of weight ranges, utility EMA settings, channel-list lengths,
  client indices, participant count, and model size.
- Added a final-phase feasibility refill so one newly infeasible client does not
  unnecessarily leave an available participant slot empty.
- Fixed multi-seed execution so an existing AirComp output cannot skip a newly
  requested M2 run. Aggregation reads only requested experiment directories,
  uses relative paths to avoid summary-name collisions, and runs cross-method
  comparison only when all three experiments were requested.
- Kept CPU process pooling limited to independent seeds. CUDA runs are serial.
  This is process-level parallelism, not parallel local client training.

The M2 fixed-budget and optional budget-sweep comparison includes the proposed
method, three ablations (`No utility`, `No upload debt`, `No starvation age`),
channel-only selection without RIS, random selection without RIS, and
channel-only selection with fixed random RIS phases. The ablations set one
corresponding priority weight to zero; the remaining weights are normalized by
the optimizer. This makes the contribution of the three scheduling signals
measurable under the same experiment setup.

## Limitations and reporting requirements

Latency, energy, propagation, and rate are simulated under configured models.
The implementation does not measure real handset CPU power or radio timing.
Results should report these as modeled values, disclose the cycles-per-sample,
CPU frequency, transmit power, bandwidth, noise, and energy assumptions, and
include a sensitivity analysis. The scheduler's validation-loss signal is a
measured held-out update utility, not a learned predictor of future client
value. Report an ablation for utility, upload debt, and starvation age, plus
random and channel-only baselines. Use matched data partitions and seeds,
multiple independent seeds, CIFAR-10 for primary evidence, and a separate small
MNIST smoke run. A short run is not publication evidence or proof of novelty.

## Dependencies

No new Python package or dataset is required for the revised M2 scheduler.
NumPy, PyTorch, PyYAML, and the existing channel/FL modules are already part of
the repository. `ProcessPoolExecutor` and spawned worker processes use Python's
standard library.
