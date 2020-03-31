from typing import Callable

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

from learn_as_you_go import BaseEmulator, emulate  # NOQA


class MeanEmulator(BaseEmulator):
    """
    An emulator that returns the mean of the training values

    The error estimate is the standard deviation of the error in the cross validation data.
    This emulator is not very useful other than as an example of how to write one.
    """

    def set_emul_func(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.emul_func: Callable[[np.ndarray], np.ndarray] = lambda x: np.mean(y_train)

    def set_emul_error_func(self, x_cv: np.ndarray, y_cv_err: np.ndarray) -> None:
        self.emul_error: Callable[[np.ndarray], np.ndarray] = lambda x: y_cv_err.std()


MEAN = 2 + np.random.uniform(size=1)


@emulate(MeanEmulator)
def noise(x: np.ndarray) -> np.ndarray:
    """
    Sample from a Gaussian distribution

    The scatter is small enough that the emulated value is always used.
    """

    return np.random.normal(loc=MEAN, scale=1e-2, size=1)


def main():
    """
    Plot some output from this emulator
    """

    NUM_TRAIN = noise.init_train_thresh
    NUM_TEST = 20
    XDIM = 1

    # Train the emulator
    x_train = np.random.uniform(size=(NUM_TRAIN, XDIM))
    y_train = np.array([noise(x) for x in x_train])

    # Output error estimates
    noise.output_err = True

    # Get values from the trained emulator
    x_emu = np.random.uniform(size=(NUM_TEST, XDIM))

    y_emu = np.zeros_like(x_emu)
    y_err = np.zeros_like(x_emu)

    for i, x in enumerate(x_emu):
        val, err = noise(x)
        y_emu[i] = val
        y_err[i] = err

    # Plot the results
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(x_train[:, 0], y_train, marker="+", label="training values")
    ax.errorbar(
        x_emu,
        y_emu,
        yerr=y_err.flatten(),
        linestyle="None",
        marker="o",
        capsize=3,
        label="emulator",
        color="red",
    )

    ax.legend()

    # `__file__` is undefined when running in sphinx
    try:
        fig.savefig(__file__ + ".png")
    except NameError:
        pass


def test_main():
    """
    Runs in pytest
    """
    main()


if __name__ == "__main__":
    main()
