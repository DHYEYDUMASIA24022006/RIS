# Reproducible Experiment Runbook

## Environment

On Windows PowerShell, from the project directory:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe main.py --experiment quick
```

The quick run is expected to finish in about half a minute on CPU. It validates
the end-to-end FL, fading, RIS, outage, metric logging, and plotting paths.

## Final-study sequence

Run each experiment separately so its logs and figures are easy to inspect:

```powershell
.\venv\Scripts\python.exe main.py --experiment baseline --seed 42
.\venv\Scripts\python.exe main.py --experiment joint --seed 42
.\venv\Scripts\python.exe main.py --experiment aircomp --seed 42
```

Repeat with seeds 7 and 2026. Before each repeat, set a distinct
`system.results_dir` in a copy of `config/config.yaml` to preserve all raw logs.
For final studies, use a CUDA device when available. CPU execution of the default
200-round CIFAR-10 configuration is intentionally slow.

## Evidence checklist

* Preserve the per-round CSV and summary JSON files.
* Record the exact configuration used for each seed.
* Build tables from the seed-level summaries; report mean plus standard deviation.
* Keep quick-test outputs separate from final-study outputs.
* If a PDF is open on Windows, the plotting helper writes a timestamped replacement
  rather than terminating the completed experiment.
