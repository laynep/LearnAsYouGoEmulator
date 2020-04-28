from __future__ import print_function

from typing import Callable

import numpy as np  # type: ignore
import scipy.optimize as opt  # type: ignore
from scipy.spatial import cKDTree as KDTree  # type: ignore

from ..util import check_good
from .emulator import BaseEmulator


class CholeskyNnEmulator(BaseEmulator):
    """
    An emulator based on Cholesky decomposition and nearest neighbours

    This emulator described in detail in arXiv:1506.01079.
    """

    def set_emul_func(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.emulator = cholesky_NN(x_train, y_train)
        self.emul_func = self.emulator.emulate

    def set_emul_error_func(self, x_cv: np.ndarray, y_cv_err: np.ndarray) -> None:
        self.emul_error: Callable[
            [np.ndarray], np.ndarray
        ] = self.emulator.train_dist_error_model(x_cv, y_cv_err)

    # TODO: override add_data


class cholesky_NN(object):
    """
    Class containing the main logic for the Cholesky and nearest neighbour based emulator
    """

    def __init__(self, xdata: np.ndarray, ydata: np.ndarray):
        """
        Initialise emulator with data

        Parameters
        ----------

        xdata : np.ndarray
            Data for the independent variable

        ydata : np.ndarray
            corresponding data for the dependent variable
        """

        # Do some tests here

        # Find data covariance
        cov = np.cov(xdata.T)

        if cov.shape == () and xdata.shape[1] == 1:
            # Handle input for a function on scalars
            # np.cov returns a scalar if xdata.T is a column vector, but
            # np.linalg.cholesky requires a 2-d array.
            # TODO: check more formally that this change makes sense
            cov = np.array([[cov]])

        # Cholesky decompose to make new basis
        L_mat = np.linalg.cholesky(cov)
        self.L_mat = np.linalg.inv(L_mat)

        # Transform xdata into new basis
        self.xtrain = xdata
        self.transf_x = np.array([np.dot(self.L_mat, x) for x in xdata])

        # DEBUG
        # import matplotlib.pyplot as plt  # type: ignore
        # plt.plot(xdata[:,0],xdata[:,1],'.',color='r')
        # plt.plot(self.transf_x[:,0],self.transf_x[:,1],'.')
        # plt.show()
        # import sys; sys.exit()

        # Store training
        self.ytrain = ydata

        # Build KDTree for quick lookup
        self.transf_xtree = KDTree(self.transf_x)

    def emulate(self, x, k=5):

        if k < 2:
            raise Exception("Need k>1")
        if x.ndim != self.xtrain[0].ndim:
            raise Exception(
                "Requested x and training set do not have the same number of dimension."
            )

        # Change basis
        x0 = np.dot(self.L_mat, x)

        # Get nearest neighbors
        dist, loc = self.transf_xtree.query(x0, k=k)
        # Protect div by zero
        dist = np.array([np.max([1e-15, d]) for d in dist])
        weight = 1.0 / dist
        nearest_y = self.ytrain[loc]

        # Interpolate with weighted average
        if self.ytrain.ndim > 1:
            y_predict = np.array([np.average(y0, weights=weight) for y0 in nearest_y.T])
            testgood = all([check_good(y) for y in y_predict])
        elif self.ytrain.ndim == 1:
            y_predict = np.average(nearest_y, weights=weight)
            testgood = check_good(y_predict)
        else:
            raise Exception("The dimension of y training data is weird")

        if not testgood:
            raise Exception("y prediction went wrong")

        return y_predict

    def train_dist_error_model(self, xtrain, ytrain, k=5):
        """Rather than learning a non-parametric error model, we can define a parametric error model instead and learn its parameters."""

        if xtrain.shape[0] != ytrain.shape[0]:
            raise TypeError("Xtrain and Ytrain do not have same shape.")

        dist_list = []
        for x0 in xtrain:

            # Change basis
            x0 = np.dot(self.L_mat, x0)

            # Get nearest neighbors in original training set
            dist, loc = self.transf_xtree.query(x0, k=k)
            # Weighted density in ball for NN
            # dist = np.array([np.max([1e-15,d]) for d in dist])
            # weight = 1.0/dist
            # dist_list.append(np.sum(weight))
            dist_list.append(np.mean(dist))

        dist_list = np.array(dist_list)
        scalar_error = np.mean(
            np.abs(ytrain), axis=0
        )  # TODO: is this a good error measure?

        def error_model(dist, a, b, c):
            return a * (dist) + b * (dist) ** c

        bestfit, cov = opt.curve_fit(
            error_model,
            dist_list,
            scalar_error,
            # bounds=((0.0,0.0,0.0),(np.inf,np.inf,np.inf)))
            bounds=((0.0, 0.0, 0.0), (1e1, 1e1, 1e1)),
        )

        # print("this is bestfit:", bestfit)

        def new_error_model(xval):
            xval = np.dot(self.L_mat, xval)
            # Get nearest neighbors in original training set
            dist, loc = self.transf_xtree.query(xval, k=k)
            # Mean distance to NN
            dist = np.mean(dist)

            # dist = dist/bestfit[2]

            err_guess = bestfit[0] * dist + bestfit[1] * dist ** bestfit[2]
            # rand_sign = np.random.rand() - 0.5
            # err_guess *= 1.0 if rand_sign>0.0 else -1.0

            return err_guess

        # DEBUG
        # import matplotlib.pyplot as plt  # type: ignore
        # plt.plot(dist_list, np.abs(ytrain),'bo')
        # plt.plot(dist_list, map(new_error_model,xtrain),'ro')
        # plt.show()

        return new_error_model
