import numpy as np
import pytest

from learn_as_you_go.emulator import emulator

DEFAULT_ABS_ERR = 0.05
DEFAULT_REL_ERR = 1.0


@pytest.mark.parametrize("xdim", [1, 2, 3])
def test_constant(xdim):
    """
    Test that a constant function can be emulated

    TODO: test that the emulatted function is called, not the true function.
    """

    CONSTANT = 1.0

    @emulator
    def constant(x):
        return CONSTANT

    for x in [np.random.uniform(size=xdim) for _ in range(1000)]:
        constant(x)

    assert constant.trained

    x = 0.5 * np.ones(xdim)

    assert np.isclose(constant(x), CONSTANT)


@pytest.mark.parametrize("xdim", [1, 2, 3])
def test_dot(xdim):
    """
    Test that a simple non-constant function can be emulated

    TODO: test that the emulated function is called, not the true function.
    """

    def true_function(x):
        return np.dot(x, x)

    @emulator
    def emulated_function(x):
        return true_function(x)

    np.random.seed(0)
    for x in [np.random.uniform(size=xdim) for _ in range(1000)]:
        emulated_function(x)

    assert emulated_function.trained

    x = 0.5 * np.ones(xdim)

    true_val = true_function(x)
    emul_val = emulated_function(x)

    assert np.abs(true_val - emul_val) < DEFAULT_ABS_ERR
    assert np.abs((emul_val - true_val) / true_val) < DEFAULT_REL_ERR


@pytest.mark.parametrize("xdim", [1, 2, 3])
def test_sin(xdim):
    """
    Test that a simple non-constant function can be emulated

    This is a sinusoid in the first dimension, constant in others.

    TODO: test that the emulated function is called, not the true function.
    """

    def true_function(x):
        return np.sin(10 * x[0])

    @emulator
    def emulated_function(x):
        return true_function(x)

    np.random.seed(0)
    for x in [np.random.uniform(size=xdim) for _ in range(1000)]:
        emulated_function(x)

    assert emulated_function.trained

    x = 0.5 * np.ones(xdim)

    true_val = true_function(x)
    emul_val = emulated_function(x)

    assert np.abs(true_val - emul_val) < DEFAULT_ABS_ERR
    assert np.abs((emul_val - true_val) / true_val) < DEFAULT_REL_ERR
