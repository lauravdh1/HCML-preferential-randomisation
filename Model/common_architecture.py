from __future__ import annotations
import sys
import warnings
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix,
)

from fairlearn.metrics import (
    MetricFrame, selection_rate, true_positive_rate, false_positive_rate,
    equalized_odds_difference, demographic_parity_ratio,
    demographic_parity_difference,
)

SEED = 2
TEST_SIZE = 0.20
CV_FOLDS = 5
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = Path(__file__).resolve().parent
CLEAN_CSV = str(PROJECT_ROOT / "data" / "h251_clean.csv")
LOG_DIR = str(MODEL_DIR / "results_log")
THRESHOLD_OBJECTIVE = "cost"
COST_FN_FP_RATIO = 2.0

RACETHX_LABELS = {1: "Hispanic", 2: "NH White", 3: "NH Black", 4: "NH Asian",
                  5: "NH Other"}

CONTINUOUS = ["AGE23X", "EDUCYR", "POVLEV23", "RTHLTH53", "MNHLTH53",
              "K6SUM42", "PHQ242", "VPCS42", "VMCS42"]
CATEGORICAL = ["MARRY23X", "REGION23", "HIDEG", "EMPST53", "ACTDTY53",
               "EVERSERVED",
               "INSCOV23", "SEX", "POVCAT23", "HAVEUS42", "TYPEPE42",
               "AFRDCA42", "AFRDPM42"]
BINARY = ["UNINS23", "ASTHDX", "HIBPDX", "ADSMOK42", "JTPAIN31_M18",
          "WLKLIM31", "STRKDX",
          "COGLIM31", "DIABDX_M18", "ACTLIM31", "CHBRON31", "ARTHDX",
          "BORNUSA", "MIDX",
          "DFHEAR42", "CANCERDX", "SOCLIM31", "DFSEE42", "OHRTDX",
          "CHOLDX", "EMPHDX",
          "CHDDX", "ANGIDX"]
FEATURES = CONTINUOUS + CATEGORICAL + BINARY
LEAKAGE = ["TOTEXP23", "TOTSLF23"]
COMORBIDITY_FLAGS = ["HIBPDX", "CHDDX", "ANGIDX", "MIDX", "OHRTDX", "STRKDX",
                     "EMPHDX",
                     "CHBRON31", "CHOLDX", "CANCERDX", "DIABDX_M18", "ARTHDX",
                     "ASTHDX"]
ENGINEERED = ["COMORBIDITY", "AGE_x_COMORBID", "COMORBID_x_UNINS"]
# engineered cols are scaled like the other continuous
CONTINUOUS = CONTINUOUS + ENGINEERED


# data
def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """ add comorbidity count + interaction terms. """
    X = X.copy()
    flags = [c for c in COMORBIDITY_FLAGS if c in X.columns]
    comorbid = X[flags].fillna(0).clip(0, 1).sum(axis=1)
    X["COMORBIDITY"] = comorbid
    X["AGE_x_COMORBID"] = X["AGE23X"] * comorbid
    if "UNINS23" in X.columns:
        X["COMORBID_x_UNINS"] = comorbid * X["UNINS23"].fillna(0)
    else:
        X["COMORBID_x_UNINS"] = 0.0
    return X


def load_clean_split(csv_path: str = CLEAN_CSV, seed: int = SEED):
    """ load the cleaned MEPS CSV and return a stratified train/test split.

    Returns unencoded feature frames; the model pipeline does the encoding so
    that scaling/one-hot are fit per CV fold (no leakage):
    X_train, X_test: feature DataFrames (45 base + 3 engineered col)
    y_train, y_test: UTILISATION (1 = >=10 visits)
    race_train, race_test: binary protected attr (1 = NH White, 0 = Non-White)
    racethx_train, racethx_test: 5-way race code (descriptive only)
    w_train, w_test: PERWT23F survey weights
    """
    df = pd.read_csv(csv_path)

    race_binary = ((df["HISPANX"] == 2) & (df["RACEV2X"] == 1)).astype(int)

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"feature col missing from {csv_path}: {missing}")

    X = engineer_features(df[FEATURES].copy())
    y = df["UTILISATION"].astype(int)
    weights = df["PERWT23F"].astype(float)
    racethx = df["RACETHX"].astype(int)

    return train_test_split(
        X, y, race_binary, racethx, weights,
        test_size=TEST_SIZE, random_state=seed, stratify=y,
    )


def build_preprocessor() -> ColumnTransformer:
    """ impute->scale continuous, impute->one-hot categorical. """
    cont = Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler())])
    cat = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore",
                                             sparse_output=False))])
    binr = SimpleImputer(strategy="constant", fill_value=0)
    return ColumnTransformer(
        transformers=[("cont", cont, CONTINUOUS),
                      ("cat", cat, CATEGORICAL),
                      ("bin", binr, BINARY)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# tuning + threshold
def tune_and_fit(pipe, grid, X_train, y_train, name=""):
    """ grid search with stratified CV, scored on ROC-AUC. """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1,
                          refit=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(X_train, y_train)
    print(
        f"[{name}] best CV ROC-AUC = {search.best_score_:.4f} "
        f"| params = {search.best_params_}"
    )
    return search.best_estimator_, search


def choose_threshold(best_pipe, X_train, y_train,
                     objective=THRESHOLD_OBJECTIVE):
    """ pick a decision threshold from OUT-OF-FOLD train predictions. """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof = cross_val_predict(best_pipe, X_train, y_train, cv=cv,
                                method="predict_proba", n_jobs=-1)[:, 1]
    grid = np.linspace(0.05, 0.95, 181)
    y = y_train.to_numpy()
    if objective == "cost":
        # minimise expected misclassification cost, weighting FN as
        # COST_FN_FP_RATIO * FP.
        costs = []
        for t in grid:
            pred = (oof >= t).astype(int)
            fn = int(np.sum((y == 1) & (pred == 0)))
            fp = int(np.sum((y == 0) & (pred == 1)))
            costs.append(COST_FN_FP_RATIO * fn + fp)
        return float(grid[int(np.argmin(costs))])
    if objective == "youden":
        scores = [recall_score(y, oof >= t) -
                  (1 - recall_score(1 - y, oof < t)) for t in grid]
    elif objective == "balanced_accuracy":
        scores = [balanced_accuracy_score(y, oof >= t) for t in grid]
    else:
        scores = [f1_score(y, oof >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


# evaluation
def overall_metrics(y_true, proba, thr, sample_weight=None):
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": round(thr, 3),
        "accuracy": accuracy_score(y_true, pred, sample_weight=sample_weight),
        "balanced_acc": balanced_accuracy_score(y_true, pred,
                                                sample_weight=sample_weight),
        "precision": precision_score(y_true, pred, zero_division=0,
                                     sample_weight=sample_weight),
        "recall_TPR": recall_score(y_true, pred, zero_division=0,
                                   sample_weight=sample_weight),
        "f1": f1_score(y_true, pred, zero_division=0,
                       sample_weight=sample_weight),
        "roc_auc": roc_auc_score(y_true, proba, sample_weight=sample_weight),
        "pr_auc": average_precision_score(y_true, proba,
                                          sample_weight=sample_weight),
        "brier": brier_score_loss(y_true, proba, sample_weight=sample_weight),
        "FN": int(fn), "FP": int(fp), "TP": int(tp), "TN": int(tn),
    }


def consistency(X_encoded, proba, n_neighbors=5):
    """ individual-fairness consistency (Zemel et al. 2013):
    Closer to 1 = similar people score similarly.
    """
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(X_encoded)
    _, idx = nn.kneighbors(X_encoded)
    idx = idx[:, 1:]
    neighbour_mean = proba[idx].mean(axis=1)
    return float(1.0 - np.mean(np.abs(proba - neighbour_mean)))


def fairness_report(model, X_test, y_test, proba, thr, race_test,
                    racethx_test):
    pred = (proba >= thr).astype(int)

    eod = equalized_odds_difference(y_test, pred,
                                    sensitive_features=race_test)
    di_ratio = demographic_parity_ratio(y_test, pred,
                                        sensitive_features=race_test)
    spd = demographic_parity_difference(y_test, pred,
                                        sensitive_features=race_test)
    mf = MetricFrame(
        metrics={"selection_rate": selection_rate, "TPR": true_positive_rate,
                 "FPR": false_positive_rate, "accuracy": accuracy_score},
        y_true=y_test, y_pred=pred, sensitive_features=race_test,
    )
    binary_df = mf.by_group.rename(index={1: "NH White (priv)",
                                          0: "Non-White"})

    X_enc = model.named_steps["prep"].transform(X_test)
    cons = consistency(np.asarray(X_enc), proba)

    g = pd.DataFrame({"racethx": racethx_test.map(RACETHX_LABELS).to_numpy(),
                      "pred": pred})
    sel = g.groupby("racethx")["pred"].mean()
    ref = sel.get("NH White", np.nan)
    racethx_df = pd.DataFrame({
        "selection_rate": sel,
        "disparate_impact_vs_NHWhite": sel / ref,
        "stat_parity_diff_vs_NHWhite": sel - ref,
    }).sort_values("selection_rate", ascending=False)

    summary = {
        "equalised_odds_diff": eod,
        "disparate_impact_ratio": di_ratio,
        "statistical_parity_diff": spd,
        "individual_fairness_consistency": cons,
    }
    return summary, binary_df, racethx_df


# loggers
class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


@contextmanager
def _tee_stdout(path):
    f = open(path, "w", encoding="utf-8")
    old = sys.stdout
    sys.stdout = _Tee(old, f)
    try:
        yield
    finally:
        sys.stdout = old
        f.close()


# run one model end to end and report to log
def evaluate_model(name, pipe, grid, csv_path: str = CLEAN_CSV,
                   log: bool = True, log_dir=LOG_DIR):

    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    log_path = Path(log_dir) / f"{safe}.log"

    @contextmanager
    def _maybe_log():
        if log:
            with _tee_stdout(log_path):
                yield
        else:
            yield

    with _maybe_log():
        if log:
            prefix = (
                "# log written "
                f"{datetime.now():%Y-%m-%d %H:%M:%S}"
                f" | seed={SEED}"
                f" | cv={CV_FOLDS}"
                f" | threshold_objective={THRESHOLD_OBJECTIVE}"
            )
            suffix = (
                f" (FN:FP={COST_FN_FP_RATIO}:1)"
                if THRESHOLD_OBJECTIVE == "cost" else ""
            )
            print(prefix + suffix)
        results = _evaluate_model_body(name, pipe, grid, csv_path)

    if log:
        print(f"[{name}] log saved to {log_path}")
    return results


# tune, pick a threshold, evaluate on the test set, print the report and
# return the results.
def _evaluate_model_body(name, pipe, grid, csv_path):
    (X_train, X_test, y_train, y_test,
     race_train, race_test, racethx_train, racethx_test,
     w_train, w_test) = load_clean_split(csv_path)

    print(f"\n ----- {name} -----")
    print(f"train={len(X_train)}  test={len(X_test)}  "
          f"features={X_train.shape[1]}  "
          f"train positive rate={y_train.mean():.3f}")

    best, _ = tune_and_fit(pipe, grid, X_train, y_train, name=name)
    thr = choose_threshold(best, X_train, y_train)
    proba = best.predict_proba(X_test)[:, 1]

    overall = overall_metrics(y_test, proba, thr)
    fair, binary_df, racethx_df = fairness_report(
        best, X_test, y_test, proba, thr, race_test, racethx_test)

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
    print("equalised_odds_diff: lower = fairer (0 = equal TPR & FPR)")
    print("disparate_impact_ratio: 1.0 = parity; < 0.80 fails the 80% rule")
    print("statistical_parity_diff: selection-rate gap vs reference group")
    print("individual_fairness_: closer to 1 = similar people score similarly")

    return {
        "name": name, "model": best, "threshold": thr, "proba": proba,
        "overall": overall, "fairness": fair,
        "binary_by_group": binary_df, "racethx_by_group": racethx_df,
        "splits": dict(X_train=X_train, X_test=X_test, y_train=y_train,
                       y_test=y_test,
                       race_train=race_train, race_test=race_test,
                       w_train=w_train, w_test=w_test),
    }
