"""
SHAP Method Implementation
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import shap

class Shap:
    
    def __init__(self, pipeline, X_train: pd.DataFrame, model_name: str, label: str):
        self.pipeline = pipeline
        self.label = label
        self.model_name = model_name

        prep = pipeline.named_steps["prep"]
        clf = pipeline.named_steps["clf"]
        X_enc = prep.transform(X_train)

        self.feature_names: list[str] = list(prep.get_feature_names_out())

        if "Logistic" in model_name:
            seld._explainer = shap.LinearExplainer(
                clf. X_enc, feature_perturbation="interventional"
            )
        else:
            self._explainer = shap.TreeExplainer(clf)
        
        self.shap_exp: shap.Explanation | None = None
        self.mean_abs: pd.Series | None = None

    def compute(self, X_test: pd.DataFrame) -> "Shap":
        """a"""
        prep = self.pipeline.named_steps["prep"]
        X_enc = prep.transform(X_test)

        raw = self._explainer.shap_values(X_enc)

        if isinstance(raw, list):
            raw = raw[1]

        base = self._explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[1]

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
        self._require_computed()
        print(f"\n-- SHAP top-{n} features | {self.label} --")
        print(self.mean_abs.sort_values(ascending=False).head(n).round(5).to_string())
