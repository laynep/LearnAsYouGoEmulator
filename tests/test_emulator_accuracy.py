import os

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pytest  # type: ignore
import torch  # type: ignore

# TODO: remove NOQA when isort is fixed
import functions  # NOQA
from learn_as_you_go import CholeskyNnEmulator  # NOQA
from learn_as_you_go import InterpolationEmulator  # NOQA
from learn_as_you_go import TorchEmulator  # NOQA

DEFAULT_ABS_ERR = 0.05
DEFAULT_REL_ERR = 1.0

CONSTANT = np.array([1.0])


@pytest.mark.parametrize(
    "true_function",
    [
        functions.true_function_constant,
        functions.true_function_identity_first_var,
        functions.true_function_dot,
        functions.true_function_sin,
        functions.true_function_exp,
        functions.true_function_step_sum,
        functions.true_function_rational,
    ],
)
@pytest.mark.parametrize("xdim", [1, 2, 3])
@pytest.mark.parametrize(
    "emulator",
    [CholeskyNnEmulator, pytest.param(TorchEmulator, marks=pytest.mark.xfail)],
)
def test_accuracy(emulator, true_function, xdim):
    """
    Test that simple functions can be emulated

    TODO: test that the emulated function is called, not the true function.
    """

    @emulator
    def emulated_function(x):
        return true_function(x)

    # Fix randoms seeds in tests
    np.random.seed(0)
    torch.manual_seed(0)

    for x in [
        np.random.uniform(size=xdim)
        for _ in range(emulated_function.initTrainThresh + 2)
    ]:
        emulated_function(x)

    assert emulated_function.trained

    emulated_function.output_err = True

    # Plot results from emulator
    num_tests = 100
    x_test = np.zeros((num_tests, xdim))
    x_test[:, 0] = np.linspace(-1, +2, num_tests)
    y_test = np.array([true_function(x) for x in x_test])
    y_emu = []
    y_err = []
    for x in x_test:
        val, err = emulated_function(x)
        y_emu.append(val)
        y_err.append(err)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    test_name = os.environ.get("PYTEST_CURRENT_TEST").split(":")[-1].split(" ")[0]
    ax.set_title(test_name)

    ax.axvspan(0, 1, label="training data range", alpha=0.2)
    ax.scatter(x_test[:, 0], y_test, marker="+", label="true value")
    ax.errorbar(
        x_test[:, 0],
        y_emu,
        yerr=y_err,
        capsize=2,
        linestyle="None",
        label="emulator",
        color="red",
    )
    ax.legend()
    # plt.show()
    fig.savefig(test_name + ".png")
    plt.close(fig)

    # Test a particular value
    x = 0.5 * np.ones(xdim)

    true_val = true_function(x)
    emul_val, _ = emulated_function(x)

    assert np.abs(true_val - emul_val) < DEFAULT_ABS_ERR
    assert np.abs((emul_val - true_val) / true_val) < DEFAULT_REL_ERR


@pytest.mark.parametrize(
    "true_function",
    [
        functions.true_function_constant,
        pytest.param(functions.true_function_dot, marks=pytest.mark.xfail),
        pytest.param(functions.true_function_sin, marks=pytest.mark.xfail),
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
