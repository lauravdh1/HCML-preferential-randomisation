import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.calibration import calibration_curve
from fairlearn.metrics import equalized_odds_difference
import common_architecture as split
from logistic_regression import build_logreg
from pathlib import Path
import numpy as np
import matplotlib
from gradient_boosting import build_gbm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
matplotlib.use("Agg")


IMAGES_DIR = split.PROJECT_ROOT / "images"
MODEL_COLORS = {"LogisticRegression": "#4C72B0", "GradientBoosting": "#DD8452"}
GROUP_ORDER = ["NH White (priv)", "Non-White"]
RACE_LABEL = {1: "NH White", 0: "Non-White"}


def _color(name, i):
    return MODEL_COLORS.get(name, plt.cm.tab10(i % 10))


def plot_roc_by_group(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                             squeeze=False)
    for ax, r in zip(axes[0], results):
        y = r["splits"]["y_test"].to_numpy()
        race = r["splits"]["race_test"].to_numpy()
        proba = r["proba"]
        for grp, ls in [(1, "-"), (0, "--")]:
            m = race == grp
            fpr, tpr, _ = roc_curve(y[m], proba[m])
            auc = roc_auc_score(y[m], proba[m])
            ax.plot(fpr, tpr, ls, label=f"{RACE_LABEL[grp]} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8, alpha=0.6)
        ax.set_title(r["name"])
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("ROC by race group", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(results, path):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, alpha=0.6,
            label="perfectly calibrated")
    for i, r in enumerate(results):
        y = r["splits"]["y_test"].to_numpy()
        prob_true, prob_pred = calibration_curve(y, r["proba"], n_bins=10,
                                                 strategy="quantile")
        ax.plot(prob_pred, prob_true, "o-", color=_color(r["name"], i),
                label=f"{r['name']} (Brier {r['overall']['brier']:.3f})")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction positive")
    ax.set_title("Calibration (reliability) curve", fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_group_rates(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(GROUP_ORDER))
    width = 0.8 / len(results)
    for metric, ax in zip(["TPR", "FPR"], axes):
        for i, r in enumerate(results):
            vals = [r["binary_by_group"].loc[g, metric] for g in GROUP_ORDER]
            ax.bar(x + i * width, vals, width, label=r["name"],
                   color=_color(r["name"], i))
        ax.set_xticks(x + width * (len(results) - 1) / 2)
        ax.set_xticklabels(GROUP_ORDER)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.set_title(f"{metric} by race group")
        ax.legend(fontsize=9)
    fig.suptitle("Group error rates (equalised-odds components)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_disparate_impact(results, path):
    order = ["NH White", "NH Black", "NH Other", "NH Asian", "Hispanic"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(order))
    width = 0.8 / len(results)
    for i, r in enumerate(results):
        di = r["racethx_by_group"]["disparate_impact_vs_NHWhite"]
        vals = [di.get(g, np.nan) for g in order]
        ax.bar(x + i * width, vals, width, label=r["name"],
               color=_color(r["name"], i))
    ax.axhline(1.0, color="grey", lw=0.8, label="parity (1.0)")
    ax.axhline(0.8, color="red", ls="--", lw=1.0, label="80% rule")
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(order)
    ax.set_ylabel("disparate impact vs NH White")
    ax.set_title("Disparate impact by race group", fontweight="bold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_threshold_tradeoff(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5),
                             squeeze=False)
    grid = np.linspace(0.05, 0.95, 91)
    for ax, r in zip(axes[0], results):
        y = r["splits"]["y_test"].to_numpy()
        race = r["splits"]["race_test"].to_numpy()
        proba = r["proba"]
        recall, eod = [], []
        for t in grid:
            pred = (proba >= t).astype(int)
            tp = np.sum((y == 1) & (pred == 1))
            fn = np.sum((y == 1) & (pred == 0))
            recall.append(tp / (tp + fn) if (tp + fn) else np.nan)
            eod.append(equalized_odds_difference(y, pred,
                                                 sensitive_features=race))
        ax.plot(grid, recall, color="#4C72B0", label="recall (TPR)")
        ax.plot(grid, eod, color="#C44E52", label="equalised-odds diff")
        ax.axvline(r["threshold"], color="grey", ls=":",
                   label=f"chosen thr={r['threshold']:.2f}")
        ax.set_title(r["name"])
        ax.set_xlabel("decision threshold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("Recall vs fairness as the threshold moves",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_all_plots(results, images_dir=IMAGES_DIR):
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("roc_by_group.png", plot_roc_by_group),
        ("calibration.png", plot_calibration),
        ("group_rates.png", plot_group_rates),
        ("disparate_impact.png", plot_disparate_impact),
        ("threshold_tradeoff.png", plot_threshold_tradeoff),
    ]
    for fname, fn in jobs:
        out = images_dir / fname
        fn(results, out)
        print(f"saved {out}")


def main():
    results = []
    for name, builder in [("LogisticRegression", build_logreg),
                          ("GradientBoosting", build_gbm)]:
        pipe, grid = builder()
        results.append(split.evaluate_model(name, pipe, grid, log=False))
    make_all_plots(results)
    print("\nall figures written to", IMAGES_DIR)


if __name__ == "__main__":
    main()
