from __future__ import annotations
import os
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.base import clone
import common_architecture as ca
from common_architecture import (
    SEED, LOG_DIR, THRESHOLD_OBJECTIVE, COST_FN_FP_RATIO, load_clean_split,
    tune_and_fit, choose_threshold, overall_metrics, fairness_report, _tee_stdout
)
from logistic_regression import build_logreg
from gradient_boosting import build_gbm


def compute_reweighing_weights(y_train: pd.Series, race_train: pd.Series) -> np.ndarray:
    """Compute the reweighing weights for each training sample (Kamiran and Calders, 2012).

    :param y_train: the binary target array-like.
    :param race_train: the binary protected attribute array-like.
    :return: the reweighing weights for each training sample.
    """
    y = np.asarray(y_train)
    a = np.asarray(race_train)
    if len(a) != len(y):
        raise ValueError("y_train and race_train must have the same length.")

    w = np.ones(len(y), dtype=float)
    for group in np.unique(a):
        for label in np.unique(y):
            mask = (a == group) & (y == label)
            p_grp = np.mean(a == group)
            p_lab = np.mean(y == label)
            p_joint = np.mean(mask)
            if p_joint > 0:
                w[mask] = (p_grp * p_lab) / p_joint
    return w


def _weight_summary(weights: np.ndarray, y_train: pd.Series, race_train: pd.Series) -> pd.DataFrame:
    """Summarises the per-cell weights for human-readable output.

    :param weights: the reweighing weights for each training sample.
    :param y_train: the binary target array-like.
    :param race_train: the binary protected attribute array-like.
    :return: a DataFrame summarising the average weight.
    """
    y = np.asarray(y_train)
    a = np.asarray(race_train)
    rows = []
    race_name = {1: "NH White (priv)", 0: "Non-White"}
    label_name = {1: "high util", 0: "low util"}
    for group in (1, 0):
        for label in (1, 0):
            mask = (a == group) & (y == label)
            if mask.sum() == 0:
                continue
            rows.append({
                "group": race_name[group],
                "label": label_name[label],
                "n": int(mask.sum()),
                "weight": round(float(weights[mask][0]), 4)
            })
    return pd.DataFrame(rows)


def evaluate_reweighing(name: str, pipe: ca.Pipeline, grid: dict, csv_path: Path = ca.CLEAN_CSV) -> dict:
    """Evaluate the reweighing mitigation method on the given pipeline and grid.

    :param name: the name of the model (for logging purposes).
    :param pipe: the sklearn Pipeline to evaluate.
    :param grid: the hyperparameter grid to search over.
    :param csv_path: the path to the cleaned CSV dataset.
    :return: a dictionary containing the overall metrics and fairness report.
    """
    (X_train, X_test, y_train, y_test,
     race_train, race_test, racethx_train, racethx_test,
     w_train, w_test) = load_clean_split(csv_path)

    print(f"\n ----- {name} (reweighing) -----")
    print(f"train={len(X_train)}  test={len(X_test)}  "
          f"features={X_train.shape[1]}  "
          f"train positive rate={y_train.mean():.3f}")

    # tune hyperparameters once on unweighted data
    best, _ = tune_and_fit(pipe, grid, X_train, y_train, w_train)
    best_params = best.get_params()

    # compute reweighing weights
    rw = compute_reweighing_weights(y_train, race_train)
    print("\n-- reweighing weights (per group x label cell) --")
    print(_weight_summary(rw, y_train, race_train).to_string(index=False))

    # refit a fresh pipeline at best params with sample weights
    rw_pipe = clone(pipe)
    rw_pipe.set_params(**{k: v for k, v in best_params.items() if k in pipe.get_params()})
    rw_pipe.fit(X_train, y_train, clf__sample_weight=rw)

    # threshold via the same rule as baseline
    thr = choose_threshold(rw_pipe, X_train, y_train)
    proba = rw_pipe.predict_proba(X_test)[:, 1]

    # evaluation
    overall = overall_metrics(y_test, proba, thr)
    fair, binary_df, racethx_df = fairness_report(
        rw_pipe, X_test, y_test, proba, thr, race_test, racethx_test
    )

    pd.set_option("display.width", 160)
    print(f"\nchosen threshold = {thr:.3f}")
    print("\n-- overall performance --")
    print(pd.Series(overall).to_string())
    print("\n-- per-group (binary race) --")
    print(binary_df.round(4).to_string())
    print("\n-- per-group (RACETHX, descriptive) --")
    print(racethx_df.round(4).to_string())
    print("\n-- fairness summary --")
    print(pd.Series(fair).round(4).to_string())

    return {
        "name": name, "model": rw_pipe, "threshold": thr, "proba": proba,
        "overall": overall, "fairness": fair,
        "binary_by_group": binary_df, "racethx_by_group": racethx_df,
        "reweighing_weights": rw,
        "splits": dict(X_train=X_train, X_test=X_test, y_train=y_train,
                       y_test=y_test, race_train=race_train,
                       race_test=race_test, w_train=w_train, w_test=w_test),
    }


def evaluate_baseline(name: str, pipe: ca.Pipeline, grid: dict, csv_path: Path = ca.CLEAN_CSV) -> dict:
    """Unmitigated run for comparison reasons."""
    return ca.evaluate_model(name, pipe, grid, csv_path)


def _delta_table(baseline, mitigated) -> pd.DataFrame:
    """Side-by-side of the metrics for comparison."""
    keys_overall = ["accuracy", "balanced_acc", "recall_TPR", "roc_auc"]
    keys_fair = ["equalised_odds_diff", "disparate_impact_ratio",
                 "statistical_parity_diff", "individual_fairness_consistency"]
    rows = []
    for k in keys_overall:
        b, m = baseline["overall"][k], mitigated["overall"][k]
        rows.append((k, b, m, m - b))
    for k in keys_fair:
        b, m = baseline["fairness"][k], mitigated["fairness"][k]
        rows.append((k, b, m, m - b))
    return pd.DataFrame(rows, columns=["metric", "baseline", "reweighed", "delta"]).round(4)


def run_one(name, builder, log=True, log_dir=LOG_DIR):
    """Full baseline + reweighing comparison for one classifier, with logging."""
    log_path = Path(log_dir) / f"{name}_reweighing.log"

    def _body():
        if log:
            print(f"# log written {datetime.now():%Y-%m-%d %H:%M:%S}"
                  f" | seed={SEED} | threshold_objective={THRESHOLD_OBJECTIVE}"
                  f" (FN:FP={COST_FN_FP_RATIO}:1) | mitigation=reweighing")

        pipe_b, grid_b = builder()
        baseline = evaluate_baseline(name, pipe_b, grid_b)

        pipe_m, grid_m = builder()
        mitigated = evaluate_reweighing(name, pipe_m, grid_m)

        print("\n========== baseline vs reweighing ==========")
        print(_delta_table(baseline, mitigated).to_string(index=False))
        print("\nReading the deltas:")
        print("  equalised_odds_diff  : lower = fairer (primary metric)")
        print("  disparate_impact_ratio: closer to 1.0 = fairer "
              "(reweighing usually moves THIS most)")
        print("  recall_TPR / accuracy : the cost we pay for fairness")
        print("  individual_fairness   : watch for drops")
        return {"baseline": baseline, "mitigated": mitigated}

    if log:
        with _tee_stdout(log_path):
            out = _body()
        print(f"[{name}] reweighing log saved to {log_path}")
    else:
        out = _body()
    return out


def main():
    results = {}
    for name, builder in [("LogisticRegression", build_logreg),
                          ("GradientBoosting", build_gbm)]:
        results[name] = run_one(name, builder, log=True)
    return results


if __name__ == "__main__":
    main()
