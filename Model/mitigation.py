from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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


# reweighing method
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
    best, _ = tune_and_fit(pipe, grid, X_train, y_train)
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


def make_aif360_dataset(X: pd.DataFrame, y: pd.Series, race: pd.Series, proba: np.ndarray) -> BinaryLabelDataset:
    """Wrap the predictions into the AIF360 post-processing dataset format.

    :param X: feature dataframe.
    :param y: binary target labels.
    :param race: binary protected attribute values for each sample.
    :param proba: predicted probability scores for each sample.
    :return: AIF360 BinaryLabelDataset with features, labels, protected attribute, and scores.
    """
    df = X.copy()
    df["UTILISATION"] = y.values
    df["RACE_BINARY"] = race.values
    df["scores"] = proba
    return BinaryLabelDataset(
        df=df,
        label_names=["UTILISATION"],
        protected_attribute_names=["RACE_BINARY"],
        scores_names=["scores"]
    )


# equalised odds method
def evaluate_cal_eqodds(name: str, pipe: ca.Pipeline, grid: dict,
                        cost_constraint: str = "fnr",
                        csv_path: Path = ca.CLEAN_CSV) -> dict:
    """Evaluate post-processing calibrated equalised odds on the given trained model.

    :param name: the name for the model to be trained.
    :param pipe: pipeline to be evaluated.
    :param grid: the hyperparameter grid for tuning.
    :param cost_constraint: which generalised cost AIF360 equalises across groups.
    :param csv_path: path to the cleaned file.
    :return: dictionary containing the overall metrics for the evaluated method.
    """
    (X_train, X_test, y_train, y_test,
     race_train, race_test, racethx_train, racethx_test,
     w_train, w_test) = load_clean_split(csv_path)

    # get the validation set from the training set -> this can be done in preprocessing PENDING
    X_tr, X_val, y_tr, y_val, race_tr, race_val = train_test_split(
        X_train, y_train, race_train, test_size=0.25, random_state=SEED, stratify=y_train
    )

    print(f"\n ----- {name} (calibrated equalised odds | cost constraint={cost_constraint}) -----")
    print(f"train={len(X_tr)}  test={len(X_test)}  "
          f"features={X_tr.shape[1]}  "
          f"train positive rate={y_tr.mean():.3f}")

    # tune hyperparameters and fit the model
    best, _ = tune_and_fit(pipe, grid, X_tr, y_tr)

    # threshold via the same rule as baseline
    thr = choose_threshold(best, X_tr, y_tr)

    # validation and test probabilities from the baseline
    proba_val = best.predict_proba(X_val)[:, 1]
    proba_test = best.predict_proba(X_test)[:, 1]

    # wrap into AIF360 datasets
    val_true = make_aif360_dataset(X_val, y_val, race_val, proba_val)
    y_val_pred = pd.Series(
        (proba_val >= thr).astype(int), index=y_val.index
    )
    val_pred = make_aif360_dataset(X_val, y_val_pred, race_val, proba_val)
    test_pred_input = make_aif360_dataset(X_test, y_test, race_test, proba_test)

    # fit the post-processor on validation
    cpp = CalibratedEqOddsPostprocessing(
        privileged_groups=[{"RACE_BINARY": 1}],
        unprivileged_groups=[{"RACE_BINARY": 0}],
        cost_constraint=cost_constraint,
        seed=SEED
    )
    # what is the predicted dataset?
    cpp.fit(val_true, val_pred)
    test_pred = cpp.predict(test_pred_input)

    # extract corrected predictions
    pred_corrected = test_pred.labels.flatten().astype(int)
    proba_corrected = test_pred.scores.flatten()

    # evaluation
    overall = overall_metrics(y_test, proba_corrected, thr, pred=pred_corrected)
    fair, binary_df, racethx_df = fairness_report(
        best, X_test, y_test, proba_corrected, thr, race_test, racethx_test,
        pred=pred_corrected
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

    print("SANITY CHECK")
    print("scores range:", proba_corrected.min(), proba_corrected.max())
    print("label rate:", pred_corrected.mean())

    # per group TPR / FPR to show which error rate the constraint actually equalised
    tpr_priv = float(binary_df.loc["NH White (priv)", "TPR"])
    tpr_unpriv = float(binary_df.loc["Non-White", "TPR"])
    fpr_priv = float(binary_df.loc["NH White (priv)", "FPR"])
    fpr_unpriv = float(binary_df.loc["Non-White", "FPR"])
    tpr_gap = abs(tpr_priv - tpr_unpriv)
    fpr_gap = abs(fpr_priv - fpr_unpriv)

    return {
        "name": name, "model": best, "threshold": thr, "proba": proba_corrected,
        "cost_constraint": cost_constraint,
        "overall": overall, "fairness": fair,
        "binary_by_group": binary_df, "racethx_by_group": racethx_df,
        "tpr_gap": tpr_gap, "fpr_gap": fpr_gap,
        "tpr_priv": tpr_priv, "tpr_unpriv": tpr_unpriv,
        "fpr_priv": fpr_priv, "fpr_unpriv": fpr_unpriv,
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
    return pd.DataFrame(rows, columns=["metric", "baseline", "mitigated", "delta"]).round(4)


def _gaps_from_result(res: dict) -> tuple[float, float]:
    """Pulls |TPR| and |FPR| group gaps from results' binary_by_group frame.

    :param res: a dictionary containing the results.
    :return: the TPR and FPR gaps extracted.
    """
    if "tpr_gap" in res and "fpr_gap" in res:
        return res["tpr_gap"], res["fpr_gap"]
    bdf = res.get("binary_by_group")
    if bdf is None:
        return float('nan'), float('nan')
    try:
        tpr_gap = abs(float(bdf.loc["NH White (priv)", "TPR"]) - float(bdf.loc["Non-White", "TPR"]))
        fpr_gap = abs(float(bdf.loc["NH White (priv)", "FPR"]) - float(bdf.loc["Non-White", "FPR"]))
    except (KeyError, TypeError):
        return float('nan'), float('nan')
    return tpr_gap, fpr_gap


def _sweep_row(label: str, res: dict) -> dict:
    """Build one row of the cost-constraint comparison table from a result dict.

    :param label: the configuration cost constraint.
    :param res: the dictionary containing the result.
    :return: a dictionary with all metrics per cost constraint.
    """
    tpr_gap, fpr_gap = _gaps_from_result(res)
    o, f = res["overall"], res["fairness"]

    # flag degenerate solutions as such
    bdf = res.get("binary_by_group")
    degenerate = False
    if bdf is not None:
        try:
            degenerate = float(bdf.loc["Non-White", "selection_rate"]) == 0.0
        except (KeyError, TypeError):
            degenerate = False

    return {
        "config": label,
        "recall_TPR": round(o["recall_TPR"], 4),
        "accuracy": round(o["accuracy"], 4),
        "balanced_acc": round(o["balanced_acc"], 4),
        "EOD": round(f["equalised_odds_diff"], 4),
        "TPR_gap": round(tpr_gap, 4),
        "FPR_gap": round(fpr_gap, 4),
        "DI_ratio": round(f["disparate_impact_ratio"], 4),
        "SPD": round(f["statistical_parity_diff"], 4),
        "indiv_fair": round(f["individual_fairness_consistency"], 4),
        "flag": "DEGENERATE" if degenerate else ""
    }


def sweep_cost_constraint(name: str, builder,
                          baseline: dict,
                          constraints: list[str] = ("fnr", "fpr", "weighted"),
                          csv_path: Path = ca.CLEAN_CSV) -> pd.DataFrame:
    """Run calibrated equalised odds under each cost constraint and build a summary table.

    :param name: name of the model used.
    :param builder: callable (pipe, grid) for each model.
    :param baseline: dictionary containint overall metrics of the baseline.
    :oaram constraints: list of all cost constraints to compare results on.
    :param csv_path: path to the cleaned data.
    :return: dataframe containing summary table.
    """
    rows = [_sweep_row("baseline (no post-proc)", baseline)]
    for cc in constraints:
        pipe, grid = builder()
        res = evaluate_cal_eqodds(name, pipe, grid, cost_constraint=cc, csv_path=csv_path)
        rows.append(_sweep_row(f"cal eq-odds [{cc}]", res))
    return pd.DataFrame(rows)


def main(method: str):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(LOG_DIR) / f"mitigation_{method}_{stamp}.log"

    def _body():
        print(f"# log written {datetime.now():%Y-%m-%d %H:%M:%S}"
              f" | seed={SEED} | threshold_objective={THRESHOLD_OBJECTIVE}"
              f" (FN:FP={COST_FN_FP_RATIO}:1) | mitigation={method}")
        results = {}

        if method == "eq_odds_sweep":
            all_tables = []
            for name, builder in [("LogisticRegression", build_logreg),
                                  ("GradientBoosting", build_gbm)]:
                pipe, grid = builder()
                pipe_b, grid_b = builder()
                baseline = evaluate_baseline(name, pipe_b, grid_b)
                table = sweep_cost_constraint(name, builder, baseline)
                table.insert(0, "model", name)
                all_tables.append(table)

                print(f"\n========== {name}: cost-constraint sweep ==========")
                print(table.to_string(index=False))

            combined = pd.concat(all_tables, ignore_index=True)
            print("\n\n########## COMBINED SUMMARY (all models x constraints) ##########")
            print(combined.to_string(index=False))
            print("\nReading the table:")
            print(" EOD: max(TPR_gap, FPR_gap); lower = fairer")
            print(" TPR_gap / FPR_gap: which error rate the constraint actually equalised")
            print(" recall_TPR: the cost paid; watch how far it drops from baseline")
            print(" DI_ratio: closer to 1.0 = fairer selection rates")
            print(" indiv_fair: closer to 1 = similar people scored similarly")
            print("\nImpossibility note: with unequal base rates, calibrated eq-odds can")
            print("equalise one error rate (the chosen constraint) only by widening the other.")
            results["combined"] = combined
            return results

        for name, builder in [("LogisticRegression", build_logreg),
                              ("GradientBoosting", build_gbm)]:
            pipe, grid = builder()
            pipe_b, grid_b = builder()
            baseline = evaluate_baseline(name, pipe_b, grid_b)

            if method == "reweighing":
                mitigated = evaluate_reweighing(name, pipe, grid)
            elif method == "eq_odds":
                mitigated = evaluate_cal_eqodds(name, pipe, grid)
            elif method == "pref_rand":
                raise NotImplementedError("Preferential randomisation not yet implemented.")
            else:
                raise ValueError(f"Unknown method: {method} Choose from: reweighing, eq_odds, pref_rand")

            print(f"\n========== baseline vs {method} ==========")
            print(_delta_table(baseline, mitigated).to_string(index=False))
            print("\nReading the deltas:")
            print(" equalised_odds_diff: lower = fairer (primary metric)")
            print(" disparate_impact_ratio: closer to 1.0 = fairer")
            print(" recall_TPR / accuracy: the cost we pay for fairness")
            print(" individual_fairness: watch for drops")
            results[name] = {"baseline": baseline, "mitigated": mitigated}
        return results

    with _tee_stdout(log_path):
        out = _body()
    print(f"[mitigation] {method} log saved to {log_path}")

    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the models with the specific fairness method."
    )
    parser.add_argument(
        "--method", type=str, required=True,
        help="Possible options are: reweighing, eq_odds, eq_odds_sweep, pref_rand"
    )
    args = parser.parse_args()

    main(args.method)
