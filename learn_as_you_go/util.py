"""
Functions that are used in various places
"""

import math

import numpy as np  # type: ignore


def check_good(x):
    """Tests if scalar is infinity, NaN, or None.

    Parameters
    ----------
    x : scalar or numpy.ndarray
        Input to test.

    Results
    -------
    good : logical
        False if x is or contains inf, NaN, or None; True otherwise.
    """

    if type(x) == np.ndarray:
        if np.all(np.isfinite(x)):
            return True
        else:
            return False

    else:
        if x == np.inf or x == -np.inf or x is None or math.isnan(x):
            return False
        else:
            return True
