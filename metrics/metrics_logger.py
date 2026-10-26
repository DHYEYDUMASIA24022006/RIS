"""Comprehensive metrics logging for FL experiments."""

import os
import csv
import json
import numpy as np

class MetricsLogger:
    """Comprehensive metrics logging for FL experiments."""
    
    METRICS = [
        'round', 'test_accuracy', 'test_loss', 'train_loss',
        'num_participating_clients', 'num_successful_uploads',
        'aggregation_mse', 'mean_snr_db', 'min_snr_db', 'max_snr_db',
        'outage_probability', 'mean_uplink_rate_bps',
        'per_round_latency_ms', 'mean_client_energy_mj',
        'jain_fairness_index', 'rounds_to_target',
        'idle_round', 'resource_constraint_violations',
        'ris_phase_norm', 'selection_strategy',
        'dirichlet_alpha', 'num_ris_elements',
        'latency_feasible_count', 'energy_feasible_count',
        'jointly_feasible_count', 'attempted_upload_count',
        'snr_rejection_count', 'empty_selection',
        'aggregation_performed', 'global_model_delta',
        'latency_is_valid', 'energy_is_valid'
    ]
    
    def __init__(self, save_dir: str, experiment_name: str):
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.history = []
        os.makedirs(self.save_dir, exist_ok=True)
        
    def log_round(self, round_num: int, **kwargs):
        """Log metrics for one round."""
        record = {'round': round_num}
        for key in self.METRICS:
            if key != 'round':
                record[key] = kwargs.get(key, None)
        self.history.append(record)
        
    def compute_jain_fairness(self, client_accuracies: list[float]) -> float:
        """J = (sum x_i)^2 / (n * sum x_i^2)."""
        if not client_accuracies:
            return 0.0
        accs = np.array(client_accuracies)
        n = len(accs)
        if n == 0 or np.sum(accs**2) == 0:
            return 0.0
        return float((np.sum(accs)**2) / (n * np.sum(accs**2)))
        
    def save_csv(self):
        """Save all logged metrics to CSV."""
        if not self.history:
            return
            
        filepath = os.path.join(self.save_dir, f"{self.experiment_name}_metrics.csv")
        with open(filepath, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.METRICS)
            writer.writeheader()
            writer.writerows(self.history)
            
    def save_json(self):
        """Save summary statistics to JSON."""
        if not self.history:
            return
            
        filepath = os.path.join(self.save_dir, f"{self.experiment_name}_summary.json")
        best_acc = max((r.get('test_accuracy') or 0.0) for r in self.history)
        final_acc = self.history[-1].get('test_accuracy', 0.0)
        
        def _mean(key):
            vals = []
            for r in self.history:
                val = r.get(key)
                if val is not None:
                    try:
                        fval = float(val)
                        if not np.isnan(fval):
                            vals.append(fval)
                    except (ValueError, TypeError):
                        pass
            return float(np.mean(vals)) if vals else None

        summary = {
            "experiment": self.experiment_name,
            "num_rounds": len(self.history),
            "best_test_accuracy": best_acc,
            "final_test_accuracy": final_acc,
            "mean_snr_db": _mean('mean_snr_db'),
            "mean_outage_probability": _mean('outage_probability'),
            "mean_successful_uploads": _mean('num_successful_uploads'),
            "mean_aggregation_mse": _mean('aggregation_mse'),
            "mean_uplink_rate_bps": _mean('mean_uplink_rate_bps'),
            "mean_latency_ms": _mean('per_round_latency_ms'),
            "mean_energy_mj": _mean('mean_client_energy_mj'),
            "final_successful_upload_fairness": self.history[-1].get('jain_fairness_index'),
            "rounds_to_target": self.history[-1].get('rounds_to_target'),
            "idle_round_fraction": _mean('idle_round'),
            "mean_resource_constraint_violations_per_round": _mean('resource_constraint_violations'),
        }
        
        with open(filepath, mode='w') as f:
            json.dump(summary, f, indent=4)
            
    def get_rounds_to_target(self, target_acc: float) -> int:
        """Return first round where accuracy >= target, or -1."""
        for record in self.history:
            acc = record.get('test_accuracy')
            if acc is not None and acc >= target_acc:
                return record['round']
        return -1
