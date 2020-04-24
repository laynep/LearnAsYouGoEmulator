"""
An example use of the `layg` package
"""

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

# TODO: remove NOQA when isort is fixed
from layg import CholeskyNnEmulator as Emulator  # NOQA
from layg import emulate  # NOQA


def main():

    ndim = 2

    ######################
    ######################
    # Toy likelihood
    @emulate(Emulator)
    def loglike(x):
        if x.ndim != 1:
            loglist = []
            for x0 in x:
                loglist.append(-np.dot(x0, x0))
            return np.array(loglist)
        else:
            return np.array(-np.dot(x, x))

    ######################
    ######################

    # Make fake data
    def get_x(ndim):

        if ndim == 1:

            return np.random.randn(1000)

        elif ndim == 2:

            return np.array([np.random.normal(0.0, 1.0), np.random.normal(0.0, 0.1)])
            # np.random.normal(1.0,0.1),
            # np.random.normal(0.0,0.1),
            # np.random.normal(0.0,60.1),
            # np.random.normal(1.0,2.1)])

        else:
            raise RuntimeError(
                "This number of dimensions has not been implemented for testing yet."
            )

    if ndim == 1:
        Xtrain = get_x(ndim)
        xlist = np.linspace(-3.0, 3.0, 11)

    elif ndim == 2:

        Xtrain = np.array([get_x(ndim) for _ in range(10000)])
        xlist = np.array([get_x(ndim) for _ in range(10)])

    else:
        raise RuntimeError(
            "This number of dimensions has not been implemented for testing yet."
        )

    Ytrain = np.array([loglike(X) for X in Xtrain])
    loglike.train(Xtrain, Ytrain)

    for x in xlist:
        print("x", x)
        print("val, err", loglike(x))

    # Plot an example
    assert loglike.trained

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x_len = 100

    x_data_plot = np.zeros((x_len, ndim))
    for i in range(ndim):
        x_data_plot[:, i] = np.linspace(0, 1, x_len)

    y_true = np.array([loglike.true_func(x) for x in x_data_plot])
    y_emul = np.array([loglike(x) for x in x_data_plot])
    y_emul_raw = np.array([loglike.emulator.emul_func(x) for x in x_data_plot])

    ax.plot(x_data_plot[..., 0], y_true, label="true", color="black")
    ax.scatter(x_data_plot[..., 0], y_emul, label="emulated", marker="+")
    ax.scatter(
        x_data_plot[..., 0],
        y_emul_raw,
        label="emulated\n no error estimation",
        marker="+",
    )

    ax.legend()

    ax.set_xlabel("Input")
    ax.set_ylabel("Output")

    fig.savefig("check.png")


def test_main():
    main()


if __name__ == "__main__":
    main()
