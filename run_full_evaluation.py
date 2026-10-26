#!/usr/bin/env python3
"""Run selected experiments across independent seeds.

Each seed has an isolated results directory. Only summaries belonging to the
requested experiments are aggregated, so stale outputs cannot contaminate a
partial rerun. CUDA runs stay serial; CPU seed runs may use spawned processes.
"""
import argparse
import copy
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
import torch
import yaml

from experiments.exp1_baseline import run_experiment as run_exp1
from experiments.exp2_joint_design import run_experiment as run_exp2
from experiments.exp3_aircomp import run_experiment as run_exp3


EXPERIMENT_DIRS = {
    "baseline": "exp1",
    "joint": "exp2",
    "aircomp": "exp3",
}


def _collect_json_summaries(seed_dir, experiments):
    """Collect summaries only from requested experiment directories."""
    result = {}
    for experiment in experiments:
        folder = os.path.join(seed_dir, EXPERIMENT_DIRS[experiment])
        if not os.path.isdir(folder):
            continue
        for root, dirs, files in os.walk(folder):
            dirs.sort()
            for filename in sorted(files):
                if not filename.endswith(".json"):
                    continue
                path = os.path.join(root, filename)
                rel = os.path.relpath(path, seed_dir).replace(os.sep, "/")
                with open(path, encoding="utf-8") as stream:
                    try:
                        result[rel] = json.load(stream)
                    except (json.JSONDecodeError, OSError) as exc:
                        raise RuntimeError(f"Could not read summary {path}: {exc}") from exc
    return result


def _flatten_numeric(value, prefix, output):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)) and np.isfinite(value):
        output.setdefault(prefix, []).append(float(value))
    elif isinstance(value, dict):
        for key, child in value.items():
            _flatten_numeric(child, f"{prefix}.{key}" if prefix else str(key), output)


def _aggregate(seed_summaries):
    keys = sorted(set().union(*(item.keys() for item in seed_summaries)))
    aggregate = {}
    for key in keys:
        flattened = {}
        for item in seed_summaries:
            if key in item:
                _flatten_numeric(item[key], "", flattened)
        stats = {field: {"mean": float(np.mean(values)),
                         "std": float(np.std(values)),
                         "values": values}
                 for field, values in flattened.items() if values}
        if stats:
            aggregate[key] = stats
    return aggregate


def _run_one_seed(config, seed, experiments, multi_dir, worker_count):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // worker_count))
    cfg = copy.deepcopy(config)
    cfg.setdefault("system", {})["seed"] = seed
    seed_dir = os.path.join(multi_dir, f"seed_{seed}")
    cfg["system"]["results_dir"] = seed_dir
    os.makedirs(seed_dir, exist_ok=True)

    # Run only what was requested. Existing output files do not imply that a
    # different requested experiment has completed.
    if "baseline" in experiments:
        run_exp1(cfg)
    if "joint" in experiments:
        run_exp2(cfg)
    if "aircomp" in experiments:
        run_exp3(cfg)

    if set(experiments) == set(EXPERIMENT_DIRS):
        from analysis.compare_methods import generate_comparison
        generate_comparison(seed_dir)
    return _collect_json_summaries(seed_dir, experiments)


def run_multi_seed(config, seeds, experiments, base_results_dir="results", workers=1):
    experiments = sorted(set(experiments) & set(EXPERIMENT_DIRS))
    if not experiments:
        raise ValueError("Select at least one experiment: baseline, joint, or aircomp")
    multi_dir = os.path.join(base_results_dir, "multi_seed")
    os.makedirs(multi_dir, exist_ok=True)
    workers = max(1, int(workers))
    requested = config.get("system", {}).get("device", "auto")
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        if workers > 1:
            print("CUDA selected: limiting multi-seed execution to one process.")
        workers = 1

    if workers == 1:
        per_seed = [_run_one_seed(config, seed, experiments, multi_dir, workers)
                    for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=get_context("spawn")) as pool:
            futures = [pool.submit(_run_one_seed, config, seed, experiments,
                                   multi_dir, workers) for seed in seeds]
            per_seed = [future.result() for future in futures]
    aggregated = _aggregate(per_seed)
    destination = os.path.join(multi_dir, "aggregated_results.json")
    with open(destination, "w", encoding="utf-8") as stream:
        json.dump({"experiments": experiments, "seeds": seeds,
                   "results": aggregated}, stream, indent=2)
    print(f"Aggregated requested experiments {experiments} across {seeds} seeds")
    print(f"Saved: {destination}")
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Multi-seed wireless FL evaluation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--experiments", nargs="+", default=["all"],
                        choices=["all", "baseline", "joint", "aircomp"])
    parser.add_argument("--workers", type=int, default=1,
                        help="Independent CPU seed processes; CUDA remains serial")
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    experiments = list(EXPERIMENT_DIRS) if "all" in args.experiments else args.experiments
    started = time.time()
    base = config.get("system", {}).get("results_dir", "results")
    run_multi_seed(config, args.seeds, experiments, base, args.workers)
    print(f"Elapsed: {(time.time() - started) / 60:.1f} minutes")


if __name__ == "__main__":
    main()
