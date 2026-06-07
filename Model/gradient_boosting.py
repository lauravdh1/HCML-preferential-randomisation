from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from common_architecture import build_preprocessor, evaluate_model, SEED
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_gbm():
    """ Histogram gradient boosting. Regularised via depth/leaves/L2/learning
    rate."""
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", HistGradientBoostingClassifier(
            random_state=SEED, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, max_iter=600)),
    ])

    # can also change to "clf__class_weight": [None, "balanced"]) for a deeper
    # search
    grid = {
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_leaf_nodes": [31, 63],
        "clf__l2_regularization": [0.0, 1.0],
        "clf__min_samples_leaf": [20],
        "clf__max_depth": [None],
    }
    return pipe, grid


if __name__ == "__main__":
    pipe, grid = build_gbm()
    results = evaluate_model("GradientBoosting", pipe, grid)
