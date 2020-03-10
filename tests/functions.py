import numpy as np  # type: ignore

CONSTANT = np.array([1.0])


def true_function_constant(x):
    return CONSTANT


def true_function_identity_first_var(x):
    return np.array([x[0]])


def true_function_dot(x):
    return np.array([np.dot(x, x)])


def true_function_sin(x):
    return np.array([np.sin(10 * x).sum()])


def true_function_exp(x):
    return np.array([np.exp(x).sum()])


def true_function_step_sum(x):
    return 1.0 + np.array([np.heaviside(x - 0.3, 0.5).sum()])


def true_function_rational(x):
    return np.array([1 + (x / (x - 1e2)).sum()])


__all__ = [
    "true_function_constant",
    "true_function_identity_first_var",
    "true_function_dot",
    "true_function_sin",
    "true_function_exp",
    "true_function_step_sum",
    "true_function_rational",
]
