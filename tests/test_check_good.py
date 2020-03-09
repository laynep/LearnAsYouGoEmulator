"""
Test whether the check_good function does what is expected

It should return False for scalars or arrays of scalars that are infinity, NaN
or None.
"""

import math

import numpy as np  # type: ignore
import pytest  # type: ignore

# TODO: remove NOQA when isort is fixed
from learn_as_you_go.util import check_good  # NOQA


@pytest.mark.parametrize("test_input", [0, 1, 1.0, np.array(1.0)])
def test_scalar_good(test_input):

    assert check_good(test_input)


@pytest.mark.parametrize(
    "test_input", [math.inf, -math.inf, math.nan, np.inf, -np.inf, np.nan, None]
)
def test_scalar_bad(test_input):

    assert not check_good(test_input)


@pytest.mark.parametrize("test_input", [0, 1, 1.0])
def test_array_good(test_input):

    array = np.zeros(10)
    array[2] = test_input

    assert check_good(test_input)


@pytest.mark.parametrize(
    "test_input", [math.inf, -math.inf, math.nan, np.inf, -np.inf, np.nan, None]
)
def test_array_bad(test_input):

    array = np.zeros(10)
    array[2] = test_input

    assert not check_good(test_input)
