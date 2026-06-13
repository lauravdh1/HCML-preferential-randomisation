"""
Preferential randomisation (Small et al., 2024) adapted to per-individual model scores.

We reuse the paper's closed-form probability curves code (equations/prob_funcs.py).
We replace the paper's population-mass optimiser with one that tallies TP/FP
over actual validation individuals, because our pipeline produces per-person
probabilities from an sklearn model rather than the paper's aggregate CreditRisk
score tables.

The objective is unchanged from the paper: per group, find thresholds (t0, t1) and
randomisation probability p that place the group's (FPR, TPR) at the shared
equalised-odds target point (FP_con, TP_con) -- the point under all groups' ROC
curves closest to (0, 1).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_curve

import equations.prob_funcs as funcs


# map a curve name to the imported closed-form function
CURVES = {
    "linear": funcs.phi,
    "quadratic": funcs.phi_quad,
    "cubic": funcs.phi_cube,
    "4th": funcs.phi_smooth,
}


def find_eo_target(proba, y, race):
    """Find the (FPR, TPR) point achievable by BOTH groups, closest to (0, 1).

    Mirrors the paper's FP_con / TP_con: the equalised-odds target that sits under
    every group's ROC curve. With two groups we walk the lower envelope of the two
    ROC curves and pick the envelope point nearest the perfect-classifier corner.

    :param proba: per-individual positive-class probabilities (validation set).
    :param y: true binary labels.
    :param race: binary protected attribute.
    :return: (fp_con, tp_con) target rates.
    """
    proba = np.asarray(proba)
    y = np.asarray(y)
    race = np.asarray(race)

    groups = np.unique(race)
    if len(groups) != 2:
        raise ValueError("This implementation assumes a binary protected attribute.")

    # ROC curve per group
    rocs = {}
    for g in groups:
        m = race == g
        fpr, tpr, _ = roc_curve(y[m], proba[m])
        rocs[g] = (fpr, tpr)

    # the point under both curves is the min of the two TPRs
    # we pick the FPR whose envelope point is closest to (0, 1).
    fpr_grid = np.linspace(0.0, 1.0, 1001)
    g0, g1 = groups
    tpr0 = np.interp(fpr_grid, rocs[g0][0], rocs[g0][1])
    tpr1 = np.interp(fpr_grid, rocs[g1][0], rocs[g1][1])
    tpr_env = np.minimum(tpr0, tpr1)  # under both curves

    dist = np.sqrt((fpr_grid - 0.0) ** 2 + (tpr_env - 1.0) ** 2)
    best = np.argmin(dist)
    return float(fpr_grid[best]), float(tpr_env[best])


def _rates_at(proba_g, y_g, t0, t1, p, curve_fn, rng):
    """Apply the curve to one group's individuals and return (FPR, TPR).

    Each individual's score yields a flip probability from the curve; we realise a
    hard decision by Bernoulli sampling so the reported rates reflect the actual
    randomised classifier.
    """
    flip_p = np.array([curve_fn(t0, t1, p, s) for s in proba_g], dtype=float)
    flip_p = np.clip(flip_p, 0.0, 1.0)
    pred = (rng.random(len(flip_p)) < flip_p).astype(int)

    y_g = np.asarray(y_g)
    tp = np.sum((pred == 1) & (y_g == 1))
    fn = np.sum((pred == 0) & (y_g == 1))
    fp = np.sum((pred == 1) & (y_g == 0))
    tn = np.sum((pred == 0) & (y_g == 0))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return fpr, tpr


def fit_group_params(proba_g, y_g, fp_con, tp_con, curve="cubic",
                     grid=21, seed=2, max_lipschitz=None):
    """Search (t0, t1, p) for one group to hit the EO target (fp_con, tp_con).

    :param max_lipschitz: if set, reject any (t0, t1, p) whose curve Lipschitz
        constant exceeds this bound.
    :return: (best_tuple, best_obj, fell_back_flag)
    """
    proba_g = np.asarray(proba_g)
    curve_fn = CURVES[curve]
    rng = np.random.default_rng(seed)

    if curve == "4th":
        p_grid = np.linspace(2 / 5, 3 / 5, grid)
    else:
        p_grid = np.linspace(0.05, 0.95, grid)

    qs = np.quantile(proba_g, np.linspace(0.02, 0.98, 25))
    thr_cands = np.unique(np.round(qs, 4))

    best = None
    best_obj = np.inf
    best_unc = None
    best_unc_obj = np.inf

    for t0 in thr_cands:
        for t1 in thr_cands:
            if t1 <= t0:
                continue
            for p in p_grid:
                fpr, tpr = _rates_at(proba_g, y_g, t0, t1, p, curve_fn, rng)
                obj = max(abs(fpr - fp_con), abs(tpr - tp_con))

                if obj < best_unc_obj:
                    best_unc_obj = obj
                    best_unc = (float(t0), float(t1), float(p), fpr, tpr)

                if max_lipschitz is not None:
                    if lipschitz_constant(t0, t1, p, curve) > max_lipschitz:
                        continue

                if obj < best_obj:
                    best_obj = obj
                    best = (float(t0), float(t1), float(p), fpr, tpr)

    if best is None:
        return best_unc, best_unc_obj, True
    return best, best_obj, False


def apply_pref_rand(proba_test, race_test, group_params, curve="cubic", seed=2):
    """Produce corrected hard predictions on the test set.

    :param proba_test: per-individual test probabilities.
    :param race_test: test protected attribute.
    :param group_params: dict {group_value: (t0, t1, p)}.
    :param curve: which curve to apply.
    :param seed: RNG seed for the Bernoulli draws.
    :return: integer array of corrected predictions.
    """
    proba_test = np.asarray(proba_test)
    race_test = np.asarray(race_test)
    curve_fn = CURVES[curve]
    rng = np.random.default_rng(seed)

    pred = np.zeros(len(proba_test), dtype=int)
    for g, (t0, t1, p) in group_params.items():
        m = race_test == g
        flip_p = np.array([curve_fn(t0, t1, p, s) for s in proba_test[m]], dtype=float)
        flip_p = np.clip(flip_p, 0.0, 1.0)
        pred[m] = (rng.random(m.sum()) < flip_p).astype(int)
    return pred


def lipschitz_constant(t0, t1, p, curve="cubic", eps=0.05):
    """Analytic-ish Lipschitz constant of the curve, per the paper's Appendix D.

    Computed the same way max_equalised_odds.py does: finite-difference the curve
    at the steepest point.
    """
    curve_fn = CURVES[curve]
    if curve == "linear":
        l1 = abs(curve_fn(t0, t1, p, t0) - curve_fn(t0, t1, p, t0 + eps)) / eps
        l2 = abs(curve_fn(t0, t1, p, t1) - curve_fn(t0, t1, p, t1 - eps)) / eps
        return max(l1, l2)
    # for quad / cubic / 4th the steepest point is at the connection tau
    xs = np.linspace(t0, t1, 500)
    vals = np.array([curve_fn(t0, t1, p, x) for x in xs])
    return float(np.max(np.abs(np.diff(vals)) / np.diff(xs)))
