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
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, EqOddsPostprocessing

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common_architecture as ca
import pref_rand as pr
from common_architecture import (
    SEED, LOG_DIR, THRESHOLD_OBJECTIVE, COST_FN_FP_RATIO, load_clean_split,
    tune_and_fit, choose_threshold, overall_metrics, fairness_report, _tee_stdout
)
from logistic_regression import build_logreg
from gradient_boosting import build_gbm
from shap_analysis import ShapAnalysis, plot_before_after


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


IMAGES_DIR = ca.PROJECT_ROOT / "images"


def _shap_before_after(name, tag, X_test, race_test,
                       model_before, Xtr_before,
                       model_after=None, Xtr_after=None,
                       model_changed=False, note_unchanged=True):
    """Run SHAP for a mitigation: summary, race-proxy and plots, before/after.

    If ``model_changed`` is True (e.g. reweighing retrains the model) the
    'before' and 'after' models are explained separately, an importance-shift
    table is printed and a before/after bar chart is saved. If False (a
    post-processing method, where the model is untouched) SHAP is computed once
    and returned as both before and after, since the attributions are identical.

    :return: (shap_before, shap_after) ShapAnalysis objects.
    """
    before = ShapAnalysis(model_before, Xtr_before, name,
                          f"{name} | {tag} before").compute(X_test)
    before.summary()
    before.race_proxy(race_test)
    before.plot(IMAGES_DIR)

    if not model_changed:
        if note_unchanged:
            print(f"\n[note] '{tag}' is post-processing: the model is "
                  f"unchanged, so SHAP attributions are identical "
                  f"before and after.")
        return before, before

    after = ShapAnalysis(model_after, Xtr_after, name,
                         f"{name} | {tag} after").compute(X_test)
    after.summary()
    after.race_proxy(race_test)
    after.plot(IMAGES_DIR)

    delta = (after.mean_abs - before.mean_abs).abs().sort_values(
        ascending=False)
    print(f"\n==== SHAP importance shift (top 10) before vs after "
          f"{tag} | {name} ====")
    print(delta.head(10).round(5).to_string())
    plot_before_after(before.mean_abs, after.mean_abs,
                      f"{name}_{tag}", IMAGES_DIR)
    return before, after


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

    # compute reweighing weights
    rw = compute_reweighing_weights(y_train, race_train)
    print("\n-- reweighing weights (per group x label cell) --")
    print(_weight_summary(rw, y_train, race_train).to_string(index=False))

    rw_pipe = clone(best)
    rw_pipe.fit(X_train, y_train, clf__sample_weight=rw)

    # shap before vs after reweighing (model retrained -> attributions change)
    shap_method_bf, shap_method_af = _shap_before_after(
        name, "reweighing", X_test, race_test,
        model_before=best, Xtr_before=X_train,
        model_after=rw_pipe, Xtr_after=X_train,
        model_changed=True)

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
        "shap_before": shap_method_bf, "shap_after": shap_method_af,
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


# calibrated equalised odds + plain equalised odds method
def _run_eqodds(name: str, pipe: ca.Pipeline, grid: dict,
                method: str,
                cost_constraint: str = "fnr",
                csv_path: Path = ca.CLEAN_CSV) -> dict:
    """Evaluate post-processing equalised odds on the given trained model.

    Shared function for both calibrated equalised odds and
    plain equalised odds (Hardt et al.).

    :param name: the name for the model to be trained.
    :param pipe: pipeline to be evaluated.
    :param grid: the hyperparameter grid for tuning.
    :param method: "calibrated" or "plain": which equalised odds method to use.
    :param cost_constraint: which generalised cost AIF360 equalises across groups
        (only applies to calibrated eqodds).
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

    if method == "calibrated":
        print(f"\n ----- {name} (calibrated equalised odds | cost constraint={cost_constraint}) -----")
    else:
        print(f"\n ----- {name} (plain equalised odds | Hardt et al.) -----")
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
    # Convert model's probabilities to yes/no predictions, otherwise
    # plain equalized odds uses true labels
    y_test_pred = pd.Series(
        (proba_test >= thr).astype(int), index=y_test.index
    )
    test_pred_input = make_aif360_dataset(X_test, y_test_pred, race_test, proba_test)

    # fit the post-processor on validation
    if method == "calibrated":
        pp = CalibratedEqOddsPostprocessing(
            privileged_groups=[{"RACE_BINARY": 1}],
            unprivileged_groups=[{"RACE_BINARY": 0}],
            cost_constraint=cost_constraint,
            seed=SEED
        )
    else:  # plain equalised odds matches both TPR and FPR
        pp = EqOddsPostprocessing(
            privileged_groups=[{"RACE_BINARY": 1}],
            unprivileged_groups=[{"RACE_BINARY": 0}],
            seed=SEED
        )
    # what is the predicted dataset?
    pp.fit(val_true, val_pred)
    test_pred = pp.predict(test_pred_input)

    # extract corrected predictions
    pred_corrected = test_pred.labels.flatten().astype(int)
    if method == "calibrated":
        proba_corrected = test_pred.scores.flatten()
    else:
        proba_corrected = proba_test

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

    # cost_constraint only applies to calibrated equalized odds
    if method == "calibrated":
        used_cost_constraint = cost_constraint
    else:
        used_cost_constraint = None

    # shap (post-processing: model unchanged -> before == after)
    tag = "cal_eqodds" if method == "calibrated" else "eq_odds_plain"
    shap_before, shap_after = _shap_before_after(
        name, tag, X_test, race_test,
        model_before=best, Xtr_before=X_tr, model_changed=False)
    shap_method = shap_before

    return {
        "name": name, "model": best, "threshold": thr, "proba": proba_corrected,
        "cost_constraint": used_cost_constraint,
        "overall": overall, "fairness": fair,
        "binary_by_group": binary_df, "racethx_by_group": racethx_df,
        "tpr_gap": tpr_gap, "fpr_gap": fpr_gap,
        "tpr_priv": tpr_priv, "tpr_unpriv": tpr_unpriv,
        "fpr_priv": fpr_priv, "fpr_unpriv": fpr_unpriv,
        "shap": shap_method,
        "splits": dict(X_train=X_train, X_test=X_test, y_train=y_train,
                       y_test=y_test, race_train=race_train,
                       race_test=race_test, w_train=w_train, w_test=w_test),
    }


# calibrated equalised odds
def evaluate_cal_eqodds(name: str, pipe: ca.Pipeline, grid: dict,
                        cost_constraint: str = "fnr",
                        csv_path: Path = ca.CLEAN_CSV) -> dict:
    """Evaluate calibrated equalised odds post-processing."""
    return _run_eqodds(name, pipe, grid, method="calibrated",
                       cost_constraint=cost_constraint, csv_path=csv_path)


# plain equalised odds (Hardt et al.)
def evaluate_eqodds(name: str, pipe: ca.Pipeline, grid: dict,
                    csv_path: Path = ca.CLEAN_CSV) -> dict:
    """Evaluate plain equalised odds post-processing (Hardt et al.)."""
    return _run_eqodds(name, pipe, grid, method="plain", csv_path=csv_path)


# preferential randomisation
def evaluate_pref_rand(name: str, pipe: ca.Pipeline, grid: dict, curve: str = "cubic",
                       max_lipschitz: float = 5.0, csv_path: str = ca.CLEAN_CSV) -> dict:
    """Evaluate the preferential randomisation (Small et al., 2024) using the
    updated version of the paper's implementation.

    :param name: name of the model.
    :param pipe: pipeline employed for the post-processing.
    :param curve: what type of ROC curve to analyse.
    :max_lipschitz: constant of the curve (similar to the paper's interpretation).
    :csv_path: path to the cleaned data.
    :return: dictionary containing all results of the post-processing.
    """
    (X_train, X_test, y_train, y_test, race_train, race_test,
     racethx_train, racethx_test, w_train, w_test) = load_clean_split(csv_path)

    X_tr, X_val, y_tr, y_val, race_tr, race_val = train_test_split(
        X_train, y_train, race_train, test_size=0.25,
        random_state=SEED, stratify=y_train)

    best, _ = tune_and_fit(pipe, grid, X_tr, y_tr)
    thr = choose_threshold(best, X_tr, y_tr)
    proba_val = best.predict_proba(X_val)[:, 1]
    proba_test = best.predict_proba(X_test)[:, 1]

    fp_con, tp_con = pr.find_eo_target(proba_val, y_val, race_val)

    group_params = {}
    for g in (0, 1):
        m = (np.asarray(race_val) == g)
        (t0, t1, p, _, _), obj, fell_back = pr.fit_group_params(
            proba_val[m], np.asarray(y_val)[m], fp_con, tp_con,
            curve=curve, grid=21, seed=SEED, max_lipschitz=max_lipschitz)
        group_params[g] = (t0, t1, p)
        print(f"group {g}: t0={t0:.3f} t1={t1:.3f} p={p:.3f} "
              f"obj={obj:.4f} L_R={pr.lipschitz_constant(t0, t1, p, curve):.3f}"
              + ("  [fell back]" if fell_back else ""))

    pred_corrected = pr.apply_pref_rand(proba_test, race_test, group_params,
                                        curve=curve, seed=SEED)

    overall = overall_metrics(y_test, proba_test, thr, pred=pred_corrected)
    fair, binary_df, racethx_df = fairness_report(
        best, X_test, y_test, proba_test, thr, race_test, racethx_test,
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

    # per group TPR / FPR to show which error rate the constraint actually equalised
    tpr_priv = float(binary_df.loc["NH White (priv)", "TPR"])
    tpr_unpriv = float(binary_df.loc["Non-White", "TPR"])
    fpr_priv = float(binary_df.loc["NH White (priv)", "FPR"])
    fpr_unpriv = float(binary_df.loc["Non-White", "FPR"])
    tpr_gap = abs(tpr_priv - tpr_unpriv)
    fpr_gap = abs(fpr_priv - fpr_unpriv)

    # shap (post-processing: model unchanged -> before == after)
    shap_before, shap_after = _shap_before_after(
        name, "pref_rand", X_test, race_test,
        model_before=best, Xtr_before=X_tr, model_changed=False)
    shap_method = shap_before

    return {
        "name": name, "model": best, "threshold": thr, "proba": proba_test,
        "overall": overall, "fairness": fair,
        "binary_by_group": binary_df, "racethx_by_group": racethx_df,
        "tpr_gap": tpr_gap, "fpr_gap": fpr_gap,
        "tpr_priv": tpr_priv, "tpr_unpriv": tpr_unpriv,
        "fpr_priv": fpr_priv, "fpr_unpriv": fpr_unpriv,
        "shap": shap_method,
        "splits": dict(X_train=X_train, X_test=X_test, y_train=y_train,
                       y_test=y_test, race_train=race_train,
                       race_test=race_test, w_train=w_train, w_test=w_test),
    }


def evaluate_baseline(name: str, pipe: ca.Pipeline, grid: dict, csv_path: Path = ca.CLEAN_CSV) -> dict:
    """Unmitigated run for comparison reasons."""
    baseline_eval = ca.evaluate_model(name, pipe, grid, csv_path)

    # shap baseline (single model explanation)
    shap_before, _ = _shap_before_after(
        name, "baseline",
        baseline_eval["splits"]["X_test"],
        baseline_eval["splits"]["race_test"],
        model_before=baseline_eval["model"],
        Xtr_before=baseline_eval["splits"]["X_train"],
        model_changed=False, note_unchanged=False)
    baseline_eval["shap"] = shap_before

    return baseline_eval


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


MODEL_COLORS = {"LogisticRegression": "#4C72B0", "GradientBoosting": "#DD8452"}
METHOD_ORDER = ["baseline", "reweighing", "cal-EO[fnr]", "plain-EO", "pref-rand"]


def _proxy_corr(shap_obj, race) -> pd.Series:
    """|correlation| between each feature's SHAP value and the race attribute."""
    vals = shap_obj.shap_exp.values
    race = np.asarray(race)
    out = {}
    for i, f in enumerate(shap_obj.feature_names):
        col = vals[:, i]
        out[f] = 0.0 if np.std(col) == 0 else abs(
            float(np.corrcoef(col, race)[0, 1]))
    return pd.Series(out)


def plot_method_comparison(df: pd.DataFrame, path: Path) -> None:
    """Two-panel results figure: EOD-by-method bars + recall-vs-EOD scatter."""
    methods = [m for m in METHOD_ORDER if m in df["method"].unique()]
    models = list(df["model"].unique())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(len(methods))
    width = 0.8 / max(len(models), 1)
    for i, mdl in enumerate(models):
        sub = df[df["model"] == mdl].set_index("method").reindex(methods)
        ax1.bar(x + i * width, sub["EOD"].values, width, label=mdl,
                color=MODEL_COLORS.get(mdl))
    ax1.axhline(0.0, color="grey", lw=0.8)
    ax1.set_xticks(x + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(methods, rotation=20, ha="right")
    ax1.set_ylabel("equalised-odds difference")
    ax1.set_title("Fairness by method (lower = fairer)", fontweight="bold")
    ax1.legend(fontsize=9)

    for mdl in models:
        sub = df[df["model"] == mdl]
        ax2.scatter(sub["EOD"], sub["recall"], s=90,
                    color=MODEL_COLORS.get(mdl), label=mdl, zorder=3)
        for _, r in sub.iterrows():
            ax2.annotate(r["method"], (r["EOD"], r["recall"]),
                         fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax2.set_xlabel("equalised-odds difference  (<- fairer)")
    ax2.set_ylabel("recall / TPR  (^ catches more high-need patients)")
    ax2.set_title("Accuracy-fairness trade-off\n(top-left = better)",
                  fontweight="bold")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path}")


def _barh(ax, series: pd.Series, title: str, color: str, n: int = 8) -> None:
    top = series.sort_values(ascending=False).head(n).iloc[::-1]
    ax.barh(range(len(top)), top.values, color=color)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=8)
    ax.set_title(title, fontsize=10)


def plot_shap_summary(shap_data: dict, path: Path) -> None:
    """One combined SHAP figure: rows = models, cols = (a) drivers, (b) race
    proxies, (c) importance shift before vs after reweighing."""
    models = list(shap_data.keys())
    fig, axes = plt.subplots(len(models), 3, figsize=(16, 4.5 * len(models)),
                             squeeze=False)
    for r, mdl in enumerate(models):
        d = shap_data[mdl]
        col = MODEL_COLORS.get(mdl)

        _barh(axes[r][0], d["drivers"], f"(a) top drivers | {mdl}", col)
        axes[r][0].set_xlabel("mean |SHAP|")

        _barh(axes[r][1], d["proxy"], f"(b) race proxies | {mdl}", "#C44E52")
        axes[r][1].set_xlabel("|corr(SHAP, race)|")

        # (c) before vs after reweighing for the top race-proxy features
        proxy_top = d["proxy"].sort_values(ascending=False).head(8).index[::-1]
        before = d["rw_before"].reindex(proxy_top).values
        after = d["rw_after"].reindex(proxy_top).values
        y = np.arange(len(proxy_top))
        h = 0.4
        ax = axes[r][2]
        ax.barh(y - h / 2, before, h, label="before", color="#4C72B0")
        ax.barh(y + h / 2, after, h, label="after", color="#DD8452")
        ax.set_yticks(y)
        ax.set_yticklabels(proxy_top, fontsize=8)
        ax.set_xlabel("mean |SHAP|")
        ax.set_title(f"(c) proxy importance: reweighing | {mdl}", fontsize=10)
        ax.legend(fontsize=8)

    fig.suptitle("SHAP: (a) prediction drivers  (b) race proxies  "
                 "(c) importance shift after reweighing",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path}")


def run_all(results: dict) -> dict:
    """Train every method for both models in one pass, print per-method delta
    tables, and write the two summary figures into images/. Returns results."""
    rows, shap_data = [], {}
    for name, builder in [("LogisticRegression", build_logreg),
                          ("GradientBoosting", build_gbm)]:
        pipe_b, grid_b = builder()
        base = evaluate_baseline(name, pipe_b, grid_b)

        pipe, grid = builder()
        rw = evaluate_reweighing(name, pipe, grid)
        pipe, grid = builder()
        cal = evaluate_cal_eqodds(name, pipe, grid, cost_constraint="fnr")
        pipe, grid = builder()
        plain = evaluate_eqodds(name, pipe, grid)

        collected = [("baseline", base), ("reweighing", rw),
                     ("cal-EO[fnr]", cal), ("plain-EO", plain)]
        try:
            pipe, grid = builder()
            pref = evaluate_pref_rand(name, pipe, grid)
            collected.append(("pref-rand", pref))
        except Exception as exc:
            print(f"[skip pref-rand for {name}]: {exc}")

        # per-method delta tables, so the consolidated log stays informative
        for label, res in collected[1:]:
            print(f"\n========== {name}: baseline vs {label} ==========")
            print(_delta_table(base, res).to_string(index=False))

        for label, res in collected:
            rows.append({
                "model": name, "method": label,
                "EOD": res["fairness"]["equalised_odds_diff"],
                "recall": res["overall"]["recall_TPR"],
                "accuracy": res["overall"]["accuracy"],
                "DI": res["fairness"]["disparate_impact_ratio"],
            })

        race = base["splits"]["race_test"]
        shap_data[name] = {
            "drivers": base["shap"].mean_abs,
            "proxy": _proxy_corr(base["shap"], race),
            "rw_before": rw["shap_before"].mean_abs,
            "rw_after": rw["shap_after"].mean_abs,
        }

    df = pd.DataFrame(rows)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    plot_method_comparison(df, IMAGES_DIR / "method_comparison.png")
    plot_shap_summary(shap_data, IMAGES_DIR / "shap_summary.png")
    print("\n########## SUMMARY METRIC TABLE ##########")
    print(df.round(4).to_string(index=False))
    results["summary"] = df
    return results


def main(method: str):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(LOG_DIR) / f"mitigation_{method}_{stamp}.log"

    def _body():
        print(f"# log written {datetime.now():%Y-%m-%d %H:%M:%S}"
              f" | seed={SEED} | threshold_objective={THRESHOLD_OBJECTIVE}"
              f" (FN:FP={COST_FN_FP_RATIO}:1) | mitigation={method}")
        results = {}

        if method == "all":
            return run_all(results)

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
            elif method == "eq_odds_calibrated":
                mitigated = evaluate_cal_eqodds(name, pipe, grid)
            elif method == "eq_odds_plain":
                mitigated = evaluate_eqodds(name, pipe, grid)
            elif method == "pref_rand":
                mitigated = evaluate_pref_rand(name, pipe, grid)
            else:
                raise ValueError(f"Unknown method: {method} Choose from: reweighing, eq_odds_calibrated, eq_odds_plain, pref_rand")

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
        help="Possible options are: reweighing, eq_odds_calibrated, eq_odds_plain, eq_odds_sweep, pref_rand, all"
    )
    args = parser.parse_args()

    main(args.method)
