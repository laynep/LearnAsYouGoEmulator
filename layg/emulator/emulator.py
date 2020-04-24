"""
Module defining the class `Emulator`, from which emulators inherit
"""
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np  # type: ignore


def raise_not_implemented_error():
    raise NotImplementedError()


class BaseEmulator(ABC):
    """
    Base class from which emulators should inherit

    This class is abstract.
    The child class must implement the marked methods.
    """

    def __init__(self):

        # The emulating emulations should not exist until the model is trained
        self.emul_func: Callable[np.ndarray, float] = raise_not_implemented_error
        self.emul_error: Callable[np.ndarray, float] = raise_not_implemented_error

    @abstractmethod
    def set_emul_func(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_emul_error_func(self, x_cv: np.ndarray, y_cv_err: np.ndarray) -> None:
        pass

    def add_data(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Add data to the training set on the fly

        By default this method does nothing.

        Overriding this method is not mandatory (some emulators require a long
        time to train, so adding a couple new points may not make sense).

        Parameters
        ----------

        x_train : np.ndarray
            Array of x-values
        y_train : np.ndarray
            Array of y-values
        """

        pass
