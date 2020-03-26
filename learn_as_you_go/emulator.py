"""
Module defining the class `Emulator`, from which emulators inherit
"""
from abc import ABC, abstractmethod

import numpy as np  # type: ignore


class BaseEmulator(ABC):
    """
    Base class from which emulators should inherit

    This class is abstract.
    The child class must implement the marked methods.
    """

    @abstractmethod
    def set_emul_func(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        pass

    @abstractmethod
    def set_emul_error_func(self, x_cv: np.ndarray, y_cv_err: np.ndarray) -> None:
        pass
