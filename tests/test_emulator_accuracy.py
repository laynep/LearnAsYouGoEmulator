import numpy as np  # type: ignore
import pytest  # type: ignore

# TODO: remove NOQA when isort is fixed
from learn_as_you_go import CholeskyNnEmulator  # NOQA
from learn_as_you_go import InterpolationEmulator  # NOQA

DEFAULT_ABS_ERR = 0.05
DEFAULT_REL_ERR = 1.0

CONSTANT = np.array([1.0])


def true_function_constant(x):
    return CONSTANT


def true_function_dot(x):
    return np.array([np.dot(x, x)])


def true_function_sin(x):
    return np.array([np.sin(10 * x[0])])


@pytest.mark.parametrize(
    "true_function", [true_function_constant, true_function_dot, true_function_sin]
)
@pytest.mark.parametrize("xdim", [1, 2, 3])
def test_cholesky(true_function, xdim):
    """
    Test that simple functions can be emulated

    TODO: test that the emulated function is called, not the true function.
    """

    @CholeskyNnEmulator
    def emulated_function(x):
        return true_function(x)

    np.random.seed(0)
    for x in [
        np.random.uniform(size=xdim)
        for _ in range(emulated_function.initTrainThresh + 2)
    ]:
        emulated_function(x)

    assert emulated_function.trained

    x = 0.5 * np.ones(xdim)

    true_val = true_function(x)
    emul_val = emulated_function(x)

    assert np.abs(true_val - emul_val) < DEFAULT_ABS_ERR
    assert np.abs((emul_val - true_val) / true_val) < DEFAULT_REL_ERR


@pytest.mark.parametrize(
    "true_function",
    [
        true_function_constant,
        pytest.param(true_function_dot, marks=pytest.mark.xfail),
        pytest.param(true_function_sin, marks=pytest.mark.xfail),
    ],
)
@pytest.mark.parametrize("xdim", [1])
def test_constant_interpolation(true_function, xdim):
    """
    Test that a constant function can be emulated

    TODO: test that the emulatted function is called, not the true function.
    """

    @InterpolationEmulator
    def emulated_function(x):
        return true_function(x)

    np.random.seed(0)
    for x in [
        np.random.uniform(size=xdim)
        for _ in range(emulated_function.initTrainThresh + 2)
    ]:
        emulated_function(x)

    assert emulated_function.trained

    x = 0.5 * np.ones(xdim)

    assert np.isclose(emulated_function(x), CONSTANT)
