import numpy as np


def phi_cube(T_0, T_1, p, score):
    x = (score - T_0) / (T_1 - T_0)
    if score >= T_1:
        return 1
    elif score < T_0:
        return 0
    elif T_0 <= score < T_0 + (T_1 - T_0) * (1 - p):
        return ((2 * p) / (p - 1) ** 3) * x ** 3 + (3 * p) / (p ** 2 - 2 * p + 1) * x ** 2
    else:
        return ((2 * (p - 1)) / (p ** 3) * x ** 3 + \
                (3 * (p ** 2 - 3 * p + 2)) / (p ** 3) * x ** 2 - \
                (6 * (p ** 2 - 2 * p + 1)) / (p ** 3) * x + \
                (p ** 3 + 3 * p ** 2 - 5 * p + 2) / (p ** 3)
                )


def phi_quad(T_0, T_1, p, score):
    x = (score - T_0) / (T_1 - T_0)
    if score >= T_1:
        return 1
    elif score < T_0:
        return 0
    elif T_0 <= score < T_0 + (T_1 - T_0) * (1 - p):
        return (p / (p - 1) ** 2) * x ** 2
    else:
        return ((p - 1) / p ** 2) * x ** 2 + (-(2 * (p - 1)) / p ** 2) * x + (p ** 2 + p - 1) / p ** 2


def phi_quad2(T_0, T_1, p, score):
    x = (score - T_0) / (T_1 - T_0)
    if score >= T_1:
        return 1
    elif score < T_0:
        return 0
    elif T_0 <= score < T_0 + (T_1 - T_0) * (1 - p):
        return (-2 * p) / (p - 1) * x - p / (p ** 2 - 2 * p + 1) * x ** 2
    else:
        return (3 * p ** 2 - 3 * p + 1) / p ** 2 - (2 * (p ** 2 - 2 * p + 1)) / p ** 2 * x - (p - 1) / p ** 2 * x ** 2


def phi(T_0, T_1, p, score):
    """
    calculate probability of not defaulting fusing piecewise linear function
    :param T_0: lower threshold
    :param T_1: upper threshold
    :param p: fixed probability of returning 1 if inbetween thresholds
    :param score: credit score
    :return: probability [0, 1]
    """
    if score >= T_1:
        return 1
    elif score < T_0:
        return 0
    elif T_0 <= score < T_0 + (T_1 - T_0) * (1 - p):
        return (p / ((T_1 - T_0) * (1 - p))) * (score - T_0)
    else:
        return ((1 - p) / ((T_1 - T_0) * p)) * (score - T_1) + 1


def phi_smooth(T_0, T_1, p, score):
    x = (score - T_0) / (T_1 - T_0)
    if score >= T_1:
        return 1
    elif score < T_0:
        return 0
    else:
        return (30 * p - 12) * x ** 2 + (-60 * p + 28) * x ** 3 + (30 * p - 15) * x ** 4


def decision(p):
    """
    return 0 or 1 determined by probability p
    :param p: probability [0, 1]
    :return: decision {0, 1}
    """
    return np.random.choice([0, 1], p=[1 - p, p])
