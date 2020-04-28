import numpy as np  # type: ignore
import pytest  # type: ignore

# TODO: remove NOQA when isort is fixed
from layg.emulator.cholesky_nn_emulator import (  # NOQA
    all_monomials,
    monomials_deg,
    n_coef,
)


@pytest.mark.parametrize(
    ["n", "delta", "true"],
    [
        # Degree 0 -> only 1
        (1, 0, 1),
        (2, 0, 1),
        (8, 0, 1),
        # Degree 1 -> n + 1
        (1, 1, 2),
        (2, 1, 3),
        (8, 1, 9),
        # Degree 2
        (1, 2, 3),
        (2, 2, 6),
    ],
)
def test_n_coef(n, delta, true):

    assert n_coef(n, delta) == true


@pytest.mark.parametrize(
    ["x", "delta", "true"],
    [
        # Degree 0 -> only 1
        (np.arange(1), 0, np.array([1])),
        (np.arange(2), 0, np.array([1])),
        (np.arange(8), 0, np.array([1])),
        # Degree 1 -> only 1
        (np.arange(1), 1, np.concatenate(([1], np.arange(1)))),
        (np.arange(2), 1, np.concatenate(([1], np.arange(2)))),
        (np.arange(8), 1, np.concatenate(([1], np.arange(8)))),
        # Degree 2
        (np.arange(1), 2, np.concatenate(([1], np.arange(1), np.arange(1) ** 2))),
    ],
)
def test_monomials(x, delta, true):

    assert np.allclose(np.sort(all_monomials(x, delta)), np.sort(true))


@pytest.mark.parametrize(
    ["x", "delta", "true"],
    [
        # Degree 1 -> linear
        (np.arange(1, 2), 1, np.array([1])),
        (np.arange(1, 3), 1, np.array([1, 2])),
        # Degree 2
        (np.arange(1, 2), 2, np.array([1])),
        (np.arange(2, 3), 2, np.array([4])),
        (np.arange(1, 3), 2, np.array([1, 2, 4])),
        (np.arange(1, 4), 2, np.array([1, 2, 3, 4, 6, 9])),
        # Degree 3
        (np.arange(2, 3), 3, np.array([8])),
        (np.arange(1, 3), 3, np.array([1, 2, 4, 8])),
        (np.arange(2, 4), 3, np.array([8, 12, 18, 27])),
    ],
)
def test_monomials_deg(x, delta, true):

    assert np.allclose(np.sort(monomials_deg(x, delta)), np.sort(true))
