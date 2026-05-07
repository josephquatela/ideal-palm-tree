"""
evaluation/conformal.py
-----------------------
Split conformal prediction calibration (Experiment 3 — Week 6).

Algorithm
---------
Given a pre-trained model and a held-out calibration set:
  1. Compute the nonconformity score for each calibration sample:
         s_i = max(y_lower_i - y_i,  y_i - y_upper_i)
     s_i > 0  →  y_i is outside the raw interval by s_i units
     s_i ≤ 0  →  y_i is inside

  2. Find δ = the ⌈(n+1)(1-α)⌉/n quantile of {s_1,...,s_n}.
     The finite-sample correction ensures coverage ≥ 1-α for exchangeable data.

  3. Expand every test interval: [ŷ_lower − δ,  ŷ_upper + δ].

One ConformalCalibrator is fitted per target so that the adjustment is in
the correct units for that target's scale.

Usage
-----
    from evaluation.conformal import ConformalCalibrator, calibrate_results

    # Predict on a held-out calibration set with any trained model
    cal = ConformalCalibrator(alpha=0.20)          # 80% target coverage
    cal.fit(y_cal, y_lower_cal, y_upper_cal)

    y_pred_adj, y_lower_adj, y_upper_adj = cal.calibrate(y_pred, y_lower, y_upper)

    # Or apply to a full results dict (one calibrator per target)
    calibrated_results = calibrate_results(raw_results, calibrators)
"""

import numpy as np


class ConformalCalibrator:
    """
    One calibrator per target.  fit() on a calibration set,
    calibrate() on any subsequent predictions.
    """

    def __init__(self, alpha: float = 0.20):
        """
        Parameters
        ----------
        alpha : float
            Miscoverage level.  alpha=0.20 targets 80% coverage.
        """
        self.alpha = alpha
        self.delta = None
        self.n_cal = None

    def fit(self, y_cal: np.ndarray, y_lower_cal: np.ndarray, y_upper_cal: np.ndarray):
        """
        Compute the conformal adjustment δ from calibration-set predictions.

        Parameters
        ----------
        y_cal       : true labels on the calibration set
        y_lower_cal : model's lower interval bound on the calibration set
        y_upper_cal : model's upper interval bound on the calibration set
        """
        scores   = np.maximum(y_lower_cal - y_cal, y_cal - y_upper_cal)
        n        = len(scores)
        # Finite-sample correction — guarantees coverage ≥ 1-alpha
        q_level  = min(np.ceil((n + 1) * (1 - self.alpha)) / n, 1.0)
        self.delta = float(np.quantile(scores, q_level, method="higher"))
        self.n_cal = n

        picp_before = float(np.mean((y_cal >= y_lower_cal) & (y_cal <= y_upper_cal)))
        print(f"  ConformalCalibrator fitted  n_cal={n}  δ={self.delta:.4f}  "
              f"PICP_before={picp_before:.3f}  target≥{1-self.alpha:.2f}")
        return self

    def calibrate(
        self,
        y_pred:  np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
    ):
        """
        Expand intervals by δ symmetrically.

        Returns (y_pred, y_lower_adj, y_upper_adj).
        y_pred is returned unchanged — calibration only affects the bounds.
        """
        if self.delta is None:
            raise RuntimeError("Call fit() before calibrate().")
        return y_pred, y_lower - self.delta, y_upper + self.delta

    @property
    def coverage_target(self):
        return 1 - self.alpha

    def __repr__(self):
        status = f"δ={self.delta:.4f}" if self.delta is not None else "unfitted"
        return f"ConformalCalibrator(alpha={self.alpha}, {status}, n_cal={self.n_cal})"


# ── Convenience helpers ───────────────────────────────────────────────────────

def fit_calibrators(
    raw_results:      dict,
    cal_df:           "pd.DataFrame",  # noqa: F821
    cal_predictions:  dict,
    alpha:            float = 0.20,
    targets: list = None,
) -> dict:
    """
    Fit one ConformalCalibrator per target.

    Parameters
    ----------
    raw_results     : results dict from any model's train_* function
                      (used for target column names only)
    cal_df          : calibration set DataFrame (with true label columns)
    cal_predictions : dict mapping target → {'y_lower': ..., 'y_upper': ...}
                      i.e. the model's raw predictions on cal_df
    alpha           : miscoverage level (default 0.20 → 80% coverage)
    targets         : list of target strings; defaults to all three

    Returns
    -------
    dict mapping target → fitted ConformalCalibrator
    """
    if targets is None:
        targets = list(raw_results.keys())

    calibrators = {}
    for target in targets:
        y_cal       = cal_df[target].values
        y_lower_cal = cal_predictions[target]["y_lower"]
        y_upper_cal = cal_predictions[target]["y_upper"]
        cal = ConformalCalibrator(alpha=alpha)
        cal.fit(y_cal, y_lower_cal, y_upper_cal)
        calibrators[target] = cal

    return calibrators


def calibrate_results(raw_results: dict, calibrators: dict) -> dict:
    """
    Apply per-target calibrators to a full results dict.

    Returns a new dict with adjusted y_lower / y_upper and updated
    calibration scores.  y_pred, y_test, mape, and model objects are
    passed through unchanged.
    """
    from evaluation.metrics import calibration_score  # local import to avoid circular

    calibrated = {}
    for target, r in raw_results.items():
        if target not in calibrators:
            calibrated[target] = r
            continue
        cal = calibrators[target]
        _, y_lower_adj, y_upper_adj = cal.calibrate(r["y_pred"], r["y_lower"], r["y_upper"])
        picp_after = calibration_score(r["y_test"], y_lower_adj, y_upper_adj)
        calibrated[target] = {
            **r,
            "y_lower":     y_lower_adj,
            "y_upper":     y_upper_adj,
            "calibration": picp_after,
        }
    return calibrated
