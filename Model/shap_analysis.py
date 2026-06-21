"""
SHAP Method Implementation
"""
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import shap
from __future__ import annotations
from pathlib import Path
matplotlib.use("Agg")


def _safe(text: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in text)


class ShapAnalysis:
    def __init__(self, pipeline, X_train: pd.DataFrame, model_name: str,
                 label: str):
        self.pipeline = pipeline
        self.label = label
        self.model_name = model_name

        prep = pipeline.named_steps["prep"]
        clf = pipeline.named_steps["clf"]
        X_enc = prep.transform(X_train)

        self.feature_names: list[str] = list(prep.get_feature_names_out())

        if "Logistic" in model_name:
            self._explainer = shap.LinearExplainer(
                clf, X_enc, feature_perturbation="interventional"
            )
        else:
            self._explainer = shap.TreeExplainer(clf)

        self.shap_exp: shap.Explanation | None = None
        self.mean_abs: pd.Series | None = None

    @staticmethod
    def _positive_class(arr):
        """Return the positive class SHAP matrix as 2D (n_samples, n_features).

        Handles every shap return shape: a list [class0, class1],
        a 3D array (n, features, classes) (new API for classifiers), or an
        already-2D array.
        """
        if isinstance(arr, list):
            arr = arr[1] if len(arr) > 1 else arr[0]
        arr = np.asarray(arr)
        if arr.ndim == 3:
            arr = arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
        return arr

    def compute(self, X_test: pd.DataFrame) -> "ShapAnalysis":
        """X_test transform and compute shap vals"""
        prep = self.pipeline.named_steps["prep"]
        X_enc = prep.transform(X_test)

        raw = self._positive_class(self._explainer.shap_values(X_enc))

        base = self._explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = np.ravel(base)
            base = float(base[1] if base.size > 1 else base[0])

        self.shap_exp = shap.Explanation(
            values=raw,
            base_values=np.full(raw.shape[0], base),
            data=X_enc,
            feature_names=self.feature_names,
        )

        self.mean_abs = pd.Series(
            np.abs(raw).mean(axis=0),
            index=self.feature_names,
            name=self.label,
        )
        return self

    def summary(self, n: int = 10) -> None:
        print(f"\n-- SHAP top-{n} features | {self.label} --")
        print(self.mean_abs.sort_values(ascending=False)
              .head(n).round(5).to_string())

    def race_proxy(self, race_test: pd.Series, n: int = 10) -> pd.Series:
        """flag features which shap correlate w race"""
        race = np.asarray(race_test)
        vals = self.shap_exp.values
        out = {}
        for i, f in enumerate(self.feature_names):
            col = vals[:, i]
            if np.std(col) == 0:
                out[f] = 0.0
            else:
                out[f] = float(np.corrcoef(col, race)[0, 1])
        correlations = pd.Series(out).abs().sort_values(ascending=False)
        print(f"\n=== shap race-proxy features (top {n}) | {self.label} ===")
        print("correlation between per-feature SHAP value and race")
        print(correlations.head(n).round(5).to_string())
        return correlations

    def plot(self, images_dir, n: int = 15) -> None:
        """Save a beeswarm and a mean-|SHAP| bar chart for this model/label."""
        images_dir = Path(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)
        safe = _safe(self.label)

        shap.plots.beeswarm(self.shap_exp, max_display=n, show=False)
        plt.title(f"SHAP beeswarm | {self.label}")
        plt.savefig(images_dir / f"shap_beeswarm_{safe}.png",
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()

        shap.plots.bar(self.shap_exp, max_display=n, show=False)
        plt.title(f"mean |SHAP| | {self.label}")
        plt.savefig(images_dir / f"shap_bar_{safe}.png",
                    dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"saved shap_beeswarm_{safe}.png, shap_bar_{safe}.png")


def plot_before_after(before: pd.Series, after: pd.Series, model_name: str,
                      images_dir, n: int = 15) -> None:
    """Grouped bar of mean-|SHAP| before vs after mitigation (top-n by before).

    Only meaningful when the model itself changed (e.g. reweighing). For
    post-processing methods the model is untouched, so before == after.
    """
    images_dir = Path(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    top = before.abs().sort_values(ascending=False).head(n).index[::-1]
    b = before.reindex(top).values
    a = after.reindex(top).values

    y = np.arange(len(top))
    height = 0.4
    fig, ax = plt.subplots(figsize=(8, max(4, 0.4 * len(top))))
    ax.barh(y - height / 2, b, height, label="before", color="#4C72B0")
    ax.barh(y + height / 2, a, height, label="after", color="#DD8452")
    ax.set_yticks(y)
    ax.set_yticklabels(top)
    ax.set_xlabel("mean |SHAP|")
    ax.set_title(f"SHAP before vs after mitigation | {model_name}",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out = images_dir / f"shap_before_after_{_safe(model_name)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out.name}")
