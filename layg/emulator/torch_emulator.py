"""
An emulator that uses a simple neural network
"""
from typing import Callable

import numpy as np  # type: ignore
from scipy.optimize import curve_fit  # type: ignore

from .emulator import BaseEmulator
from .nn import Net


class TorchEmulator(BaseEmulator):
    """
    Class that uses pytorch to do emulation

    The Universal Approximation Theorem says that any Lebesgue integrable
    function can be approximated by a feed-forward network with sufficient
    layers of sufficient width.  It doesn't guarantee that we can train the
    network though.
    """

    def set_emul_func(self, x_train: np.ndarray, y_train: np.ndarray) -> None:

        self.x_train = x_train
        self.y_train = y_train

        n_input = x_train.shape[1]
        n_output = y_train.shape[1]
        n_hidden = 20

        # Define network with appropriate size
        self.net = Net(n_input, n_hidden, n_output)

        # Set network mode to training
        self.net.train()

        # Train network on given data
        self.net.my_train(x_train, y_train, 5000)

        # Set network mode to evaluation
        self.net.eval()

        self.emul_func = self.net.call_numpy

    def set_emul_error_func(self, x_cv: np.ndarray, y_cv_err: np.ndarray) -> None:
        """
        Fit a quadratic to the residuals and mean distance to nearby points
        """

        num_neighbors = 2 * x_cv.shape[1]

        dist_list = np.empty(x_cv.shape[0])
        # y_pred = np.empty(y_cv.shape)
        for i, x in enumerate(x_cv):
            distance_array = ((self.x_train - x) ** -2).sum(axis=1) ** -0.5
            dist_list[i] = np.mean(np.sort(distance_array)[num_neighbors:])
            # y_pred[i] = self.net.call_numpy(x)

        # TODO: choose another function to scalarify errors
        errors = (y_cv_err ** 2).sum(axis=1) ** 0.5

        def error_model(dist, a, b, c):
            """
            The error model function to be fit
            """

            return a + b * dist + c * dist ** 2

        popt, _ = curve_fit(
            error_model, dist_list, errors, bounds=(np.zeros(3), 1 / np.zeros(3))
        )

        def error(x):

            distance_array = ((self.x_train - x) ** -2).sum(axis=1) ** -0.5
            distance = np.mean(np.sort(distance_array)[num_neighbors:])

            return error_model(distance, *popt)

        self.emul_error: Callable[[np.ndarray], np.ndarray] = error
