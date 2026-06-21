"""
SHAP Method Implementation
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import shap

class ShapAnalysis:
    def __init__(self, pipeline, X_train: pd.DataFrame, model_name: str, label: str):
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

    def compute(self, X_test: pd.DataFrame) -> "ShapAnalysis":
        """X_test transform and compute shap vals"""
        prep = self.pipeline.named_steps["prep"]
        X_enc = prep.transform(X_test)

        raw = self._explainer.shap_values(X_enc)

        if isinstance(raw, list):
            raw = raw[1]

        base = self._explainer.expected_value
        if isinstance(base, (list, np.ndarray)) and len(base) > 1:
            base = base[1]
        elif isinstance(base, (list, np.ndarray)):
            base = base[0]

        self.shap_exp = shap.Explanation(
            values=raw,
            base_values=base,
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
        print(self.mean_abs.sort_values(ascending=False).head(n).round(5).to_string())

    def race_proxy(self, race_test: pd.Series, n: int = 10) -> None:
        """flag features which shap correlate w race"""
        race = np.asarray(race_test)
        correlations = pd.Series({f: float(np.corrcoef(self.shap_exp.values[:, i], race)[0, 1]) 
            for i, f in enumerate(self.feature_names)})
        correlations = correlations.abs().sort_values(ascending=False)

        print(f"\n=======shap race-proxy features (top {n}) | {self.label}=======")
        print("\n correlation ==== shap & race attribute")
        print(correlations.head(n).round(5).to_string())

