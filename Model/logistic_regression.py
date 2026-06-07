from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from common_architecture import build_preprocessor, evaluate_model, SEED
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_logreg():
    """ L2 logistic regression. Tunes C (inverse reg. strength) and
    class_weight."""
    pipe = Pipeline([
        ("prep", build_preprocessor()),
        ("clf", LogisticRegression(penalty="l2", solver="liblinear",
                                   max_iter=2000, random_state=SEED)),
    ])
    grid = {
        "clf__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
        "clf__class_weight": [None],
    }
    return pipe, grid


if __name__ == "__main__":
    pipe, grid = build_logreg()
    results = evaluate_model("LogisticRegression", pipe, grid)
