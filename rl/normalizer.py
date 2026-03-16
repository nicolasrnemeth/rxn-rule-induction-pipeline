# import math
# from typing import Dict

# class RunningMetricNormalizer:
#     """
#     Welford running mean/std per metric for variance reduction.
#     """
#     def __init__(self):
#         self.n = {}
#         self.mean = {}
#         self.m2 = {}

#     def update(self, metrics: Dict[str, float]):
#         for k, v in metrics.items():
#             n_old = self.n.get(k, 0)
#             mean_old = self.mean.get(k, 0.0)
#             m2_old = self.m2.get(k, 0.0)

#             n_new = n_old + 1
#             delta = v - mean_old
#             mean_new = mean_old + delta / n_new
#             m2_new = m2_old + delta * (v - mean_new)

#             self.n[k] = n_new
#             self.mean[k] = mean_new
#             self.m2[k] = m2_new

#     def normalize(self, metrics: Dict[str, float], eps: float = 1e-6) -> Dict[str, float]:
#         out = {}
#         for k, v in metrics.items():
#             n = self.n.get(k, 0)
#             if n < 2:
#                 out[k] = 0.0
#             else:
#                 var = self.m2[k] / (n - 1)
#                 std = math.sqrt(max(var, eps))
#                 out[k] = (v - self.mean[k]) / std
#         return out

import math
from typing import Dict, Optional

class RunningMetricNormalizer:
    """
    Running normalization for RL metrics with configurable strategy.
    Supports:
      - Welford (equal weighting)
      - EMA (exponential decay)
      - Warmup + freeze
      - Clipping
      - Per-metric overrides (e.g., treat bounded metrics differently)
    """

    def __init__(
        self,
        use_ema: bool = False,
        ema_beta: float = 0.99,
        warmup: int = 0,
        freeze_after_warmup: bool = True,
        clip: Optional[float] = 3.0,
        std_floor: float = 1e-3,
        treat_bounded_as_center_only: bool = True,
        bounded_metrics: Optional[Dict[str, bool]] = None,
    ):
        self.use_ema = use_ema
        self.ema_beta = ema_beta
        self.warmup = warmup
        self.freeze_after_warmup = freeze_after_warmup
        self.clip = clip
        self.std_floor = std_floor
        self.treat_bounded_as_center_only = treat_bounded_as_center_only
        self.bounded_metrics = bounded_metrics or {}
        self._frozen = False

        # Welford states
        self.n = {}
        self.mean = {}
        self.m2 = {}

        # EMA states (if enabled)
        self.ema_mean = {}
        self.ema_var = {}

        self.updates = 0

    def update(self, metrics: Dict[str, float]):
        """
        Update running stats. If warmup with freeze is enabled and warmup reached, freeze stats.
        """
        if self._frozen:
            return

        self.updates += 1
        for k, v in metrics.items():
            # Welford
            n_old = self.n.get(k, 0)
            mean_old = self.mean.get(k, 0.0)
            m2_old = self.m2.get(k, 0.0)

            n_new = n_old + 1
            delta = v - mean_old
            mean_new = mean_old + delta / n_new
            m2_new = m2_old + delta * (v - mean_new)

            self.n[k] = n_new
            self.mean[k] = mean_new
            self.m2[k] = m2_new

            if self.use_ema:
                # EMA
                ema_mean_old = self.ema_mean.get(k, v)
                ema_var_old = self.ema_var.get(k, 0.0)
                ema_mean_new = self.ema_beta * ema_mean_old + (1 - self.ema_beta) * v
                # For variance: use (x - mean_old)^2, then update
                diff = v - ema_mean_old
                ema_var_new = self.ema_beta * ema_var_old + (1 - self.ema_beta) * (diff * diff)

                self.ema_mean[k] = ema_mean_new
                self.ema_var[k] = ema_var_new

        if self.warmup > 0 and self.freeze_after_warmup and self.updates >= self.warmup:
            self._frozen = True

    def _get_stats(self, k: str):
        if self.use_ema:
            mean = self.ema_mean.get(k, 0.0)
            var = self.ema_var.get(k, 0.0)
        else:
            n = self.n.get(k, 0)
            if n < 2:
                return 0.0, 0.0
            var = self.m2[k] / (n - 1)
            mean = self.mean[k]
        return mean, var

    def normalize(self, metrics: Dict[str, float], eps: float = 1e-6) -> Dict[str, float]:
        """
        Normalize per metric.
        For bounded metrics (if flagged) and treat_bounded_as_center_only=True:
            normalized = (v - mean)
        Else:
            z-score with floor and clipping.
        """
        out = {}
        for k, v in metrics.items():
            mean, var = self._get_stats(k)
            if var <= 0.0:
                out[k] = 0.0
                continue

            std = math.sqrt(max(var, eps))
            use_center_only = self.treat_bounded_as_center_only and self.bounded_metrics.get(k, False)

            if use_center_only:
                z = v - mean
            else:
                if std < self.std_floor:
                    # Avoid huge values when variance collapses
                    z = (v - mean) / self.std_floor
                else:
                    z = (v - mean) / std

            if self.clip is not None:
                z = max(-self.clip, min(self.clip, z))

            out[k] = z
        return out

    def is_frozen(self) -> bool:
        return self._frozen

    def reset(self):
        self.n.clear()
        self.mean.clear()
        self.m2.clear()
        self.ema_mean.clear()
        self.ema_var.clear()
        self.updates = 0
        self._frozen = False

    def current_stats(self) -> Dict[str, Dict[str, float]]:
        stats = {}
        for k in self.mean.keys():
            mean, var = self._get_stats(k)
            stats[k] = {
                "mean": mean,
                "std": math.sqrt(var) if var > 0 else 0.0,
                "n": self.n.get(k, 0),
            }
        return stats