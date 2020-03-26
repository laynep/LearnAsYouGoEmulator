from __future__ import print_function

from typing import Callable

import numpy as np  # type: ignore
import scipy.interpolate as interp  # type: ignore

from .emulator import BaseEmulator


class InterpolationEmulator(BaseEmulator):
    """
    An emulator based on interpolation

    This emulator uses `scipy.interpolate.interp1d` to interpolate functions
    with scalar image and preimage.

    It returns infinity for values outside the trained domain.
    TODO: determine whether an exception should be raised instead. Then the
    emulator should evaluate the true function.
    """

    def interpolator(self, xdata: np.ndarray, ydata: np.ndarray):

        if ydata.shape[1] > 1:
            raise TypeError(
                "Cannot interpolate when range has higher dimensions than 1."
            )

        if xdata.shape[1] > 1:
            raise TypeError(
                "The interpolator is not yet set up for higher dimensions than ",
                xdata.ndim - 1,
            )

        if xdata.shape[0] != ydata.shape[0]:
            raise TypeError("The x and y data do not have the same number of elements.")

        # Reshape data from Nx1 matrix to N-vector
        xdata = xdata.T[0]
        ydata = ydata.T[0]

        interp_funct = interp.interp1d(xdata, ydata)
        xmin = np.min(xdata)
        xmax = np.max(xdata)

        def predict(x):
            # Reshape data from 1x1 matrix to scalar
            x = x.T[0]

            if x < xmin or x > xmax:
                pred = np.inf
            else:
                pred = interp_funct(x)
            return pred

        return predict

    def set_emul_func(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.emul_func: Callable[[np.ndarray], np.ndarray] = self.interpolator(
            x_train, y_train
        )

    def set_emul_error_func(self, x_cv: np.ndarray, y_cv_err: np.ndarray) -> None:
        """
        TODO: This error estimate is pretty bad; it fits the error as a function of x.
        Should be as a function of distance to nearest point.
        """
        self.emul_error: Callable[[np.ndarray], np.ndarray] = self.interpolator(
            x_cv, y_cv_err
        )
