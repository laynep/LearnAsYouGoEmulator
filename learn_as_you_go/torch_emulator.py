"""
An emulator that uses a simple neural network
"""

import numpy as np  # type: ignore

from .emulator import BaseEmulator
from .learner import Learner
from .nn import Net


class TorchEmulator(Learner, BaseEmulator):
    """
    Class that uses pytorch to do emulation

    The Universal Approximation Theorem says that any Lebesgue integrable
    function can be approximated by a feed-forward network with sufficient
    layers of sufficient width.  It doesn't guarantee that we can train the
    network though.
    """

    def set_emul_func(self, x_train: np.ndarray, y_train: np.ndarray) -> None:

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

    def set_emul_error_func(self, x_train: np.ndarray, y_train: np.ndarray) -> None:

        # TODO set error model
        self.emul_error = lambda x: 0.0
