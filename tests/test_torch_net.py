import os

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pytest  # type: ignore
import torch  # type: ignore

# TODO: remove NOQA when isort is fixed
import functions  # NOQA
from learn_as_you_go.emulator.torch_emulator import Net  # NOQA

DEFAULT_ABS_ERR = 0.05
DEFAULT_REL_ERR = 1.0


@pytest.mark.parametrize(
    "true_function",
    [
        functions.true_function_constant,
        functions.true_function_identity_first_var,
        functions.true_function_dot,
        pytest.param(functions.true_function_sin, marks=pytest.mark.xfail),
    ],
)
@pytest.mark.parametrize("xdim", [1])
# @pytest.mark.parametrize("xdim", [1, 2, 3])
def test_neural_network_accuracy(true_function, xdim):
    """
    Test that simple functions can be emulated

    TODO: test that the emulated function is called, not the true function.
    """

    num_epochs = 5000
    num_examples = 1000
    num_tests = 100

    test_name = os.environ.get("PYTEST_CURRENT_TEST").split(":")[-1].split(" ")[0]

    # Fix randoms seeds in tests
    np.random.seed(0)
    torch.manual_seed(0)

    x_train = np.random.uniform(-1, 1, (num_examples, 1))
    y_train = np.array([true_function(x) for x in x_train])

    x_test = np.reshape(
        np.linspace(x_train.min() - 1, x_train.max() + 1, num_tests), (num_tests, 1)
    )
    y_test = np.array([true_function(x) for x in x_test])

    net = Net(xdim, 20, 1)

    print("Initial: ", net.call_numpy(np.array([0.4])))
    net.my_train(np.array(x_train), np.array(y_train), num_epochs)
    print("Final: ", net.call_numpy(np.array([0.4])))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(test_name)

    ax.scatter(x_train, y_train, marker="x", label="training data")
    ax.scatter(x_test, y_test, marker="+", label="test data")
    ax.plot(x_test, [net.call_numpy(x) for x in x_test], label="net", color="red")
    ax.legend()
    # plt.show()

    fig.savefig(test_name + ".png")

    # Test a particular value
    x = 0.5 * np.ones(xdim)
    true_val = true_function(x)
    emul_val = net.call_numpy(x)

    assert np.abs(true_val - emul_val) < DEFAULT_ABS_ERR
    assert np.abs((emul_val - true_val) / true_val) < DEFAULT_REL_ERR
