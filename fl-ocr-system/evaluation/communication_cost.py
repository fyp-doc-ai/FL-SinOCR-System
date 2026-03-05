"""Communication cost tracking for federated learning experiments."""

from typing import Dict, List

import numpy as np
from torch import nn

from models.model_utils import compute_parameter_bytes


class CommunicationTracker:
    """Tracks communication costs across FL rounds.

    Measures bytes transferred per round (upload + download) and
    cumulative totals.
    """

    def __init__(self, model: nn.Module, trainable_only: bool = True):
        self.bytes_per_param_set = compute_parameter_bytes(model, trainable_only)
        self.round_logs: List[Dict[str, float]] = []

    def log_round(
        self,
        server_round: int,
        num_clients_fit: int,
        num_clients_eval: int = 0,
    ) -> Dict[str, float]:
        """Log communication cost for a single round.

        Each fitting client:
          - downloads global model params (bytes_per_param_set)
          - uploads local model params (bytes_per_param_set)
        Each evaluating client:
          - downloads global model params only
        """
        download_bytes = (num_clients_fit + num_clients_eval) * self.bytes_per_param_set
        upload_bytes = num_clients_fit * self.bytes_per_param_set
        round_total = download_bytes + upload_bytes

        cumulative = sum(r["round_total_bytes"] for r in self.round_logs) + round_total

        entry = {
            "round": server_round,
            "num_clients_fit": num_clients_fit,
            "num_clients_eval": num_clients_eval,
            "download_bytes": download_bytes,
            "upload_bytes": upload_bytes,
            "round_total_bytes": round_total,
            "cumulative_bytes": cumulative,
            "round_total_mb": round_total / (1024 * 1024),
            "cumulative_mb": cumulative / (1024 * 1024),
        }
        self.round_logs.append(entry)
        return entry

    def get_summary(self) -> Dict[str, float]:
        """Return overall communication summary."""
        if not self.round_logs:
            return {"total_bytes": 0, "total_mb": 0, "num_rounds": 0}

        total = self.round_logs[-1]["cumulative_bytes"]
        return {
            "total_bytes": total,
            "total_mb": total / (1024 * 1024),
            "total_gb": total / (1024 * 1024 * 1024),
            "num_rounds": len(self.round_logs),
            "avg_bytes_per_round": total / len(self.round_logs),
            "bytes_per_param_set": self.bytes_per_param_set,
        }

    def get_all_round_logs(self) -> List[Dict[str, float]]:
        """Return all per-round communication logs."""
        return self.round_logs
