import os

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pytest  # type: ignore
import torch  # type: ignore

# TODO: remove NOQA when isort is fixed
import functions  # NOQA
from layg import CholeskyNnEmulator  # NOQA
from layg import InterpolationEmulator  # NOQA
from layg import TorchEmulator  # NOQA
from layg import emulate  # NOQA

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
    "emulator", [CholeskyNnEmulator, TorchEmulator],
)
def test_accuracy(emulator, true_function, xdim):
    """
    Test that simple functions can be emulated

    TODO: test that the emulated function is called, not the true function.
    """

    @emulate(emulator)
    def emulated_function(x):
        return true_function(x)

    # Fix randoms seeds in tests
    np.random.seed(0)
    torch.manual_seed(0)

    for x in [
        np.random.uniform(size=xdim)
        for _ in range(emulated_function.init_train_thresh + 2)
    ]:
        emulated_function(x)

    assert emulated_function.trained

    emulated_function.output_err = True

    # Plot results from emulator
    num_tests = 100
    x_test = np.zeros((num_tests, xdim))
    x_test[:, 0] = np.linspace(-1, +2, num_tests)
    y_test = np.array([true_function(x) for x in x_test])
    y_emu = np.empty_like(y_test)
    y_err = np.empty(num_tests)
    for i, x in enumerate(x_test):
        val, err = emulated_function(x)
        y_emu[i] = val
        y_err[i] = err

    # Project data
    x_true = x_test[:, 0]
    y_true = y_test[:, 0]
    x_exact = x_true[y_err == 0.0]
    y_exact = y_emu[y_err == 0.0]
    y_err_exact = y_err[y_err == 0.0]
    x_emulated = x_true[y_err != 0.0]
    y_emulated = y_emu[y_err != 0.0]
    y_err_emulated = y_err[y_err != 0.0]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    test_name = os.environ.get("PYTEST_CURRENT_TEST").split(":")[-1].split(" ")[0]
    ax.set_title(test_name)

    ax.axvspan(0, 1, label="training data range", alpha=0.2)
    ax.scatter(x_true, y_true, marker="+", label="true value")
    ax.errorbar(
        x_exact,
        y_exact,
        yerr=y_err_exact,
        capsize=2,
        linestyle="None",
        label="exact",
        color="black",
    )
    ax.errorbar(
        x_emulated,
        y_emulated,
        yerr=y_err_emulated,
        capsize=2,
        linestyle="None",
        label="emulated",
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

    assert np.abs(true_val - emul_val) < emulated_function.abs_err_local
    assert np.abs((emul_val - true_val) / true_val) < emulated_function.frac_err_local


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

    @emulate(InterpolationEmulator)
    def emulated_function(x):
        return true_function(x)

    np.random.seed(0)
    for x in [
        np.random.uniform(size=xdim)
        for _ in range(emulated_function.init_train_thresh + 2)
    ]:
        emulated_function(x)

    assert emulated_function.trained

    x = 0.5 * np.ones(xdim)

    assert np.isclose(emulated_function(x), CONSTANT)
