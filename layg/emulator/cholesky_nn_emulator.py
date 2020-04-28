from __future__ import print_function

import warnings
from itertools import product
from typing import Callable

import numpy as np  # type: ignore
import scipy.optimize as opt  # type: ignore
from scipy import special  # type: ignore
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

    def emulate(self, x, delta=2):

        if delta < 0 or not isinstance(delta, int):
            raise ValueError(
                "Got degree of interpolation {}, must be a non-negative integer".format(
                    delta
                )
            )
        if x.ndim != self.xtrain[0].ndim:
            raise Exception(
                "Requested x and training set do not have the same number of dimension."
            )

        # Change basis
        x0 = np.dot(self.L_mat, x)

        # Need enough neighbours to constrain fit with degree delta polynomials
        k = 2 * n_coef(x.shape[0], delta)

        # Get nearest neighbors
        dist, loc = self.transf_xtree.query(x0, k=k)
        nearest_x = self.transf_x[loc]
        nearest_y = self.ytrain[loc]

        # Protect div by zero
        dist = np.array([np.max([1e-15, d]) for d in dist])
        weight = 1.0 / dist

        try:
            y_predict = polynomial_fit(x0, nearest_x, nearest_y, weight, delta)
        except np.linalg.LinAlgError as lin_alg_error:
            # TODO: this is a hack; should propagate exceptions carefully
            warnings.warn("Exception inside emulator: {}".format(lin_alg_error))

            # return nans and let caller deal with it
            output = np.empty(self.ytrain[0].shape[0])
            output.fill(np.nan)
            return output

        if not check_good(y_predict):
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


def polynomial_fit(
    x_0: np.ndarray, x_nn: np.ndarray, y_nn: np.ndarray, weights: np.ndarray, delta: int
) -> np.ndarray:
    """
    Compute polynomial fit

    Assumes that the point of interest is at the origin

    Parameters
    ----------

    x_0 : np.ndarray
        point at which to evaluate polynomial fit

    x_nn : np.ndarray
        distances to nearest neighbours

    y_nn : np.ndarray
        y-values at nearest neighbours

    delta : int
        degree of polynomial
    """

    n = x_0.shape[0]
    k = x_nn.shape[0]
    m = y_nn.shape[1]

    # Check that shapes are consistent
    assert x_nn.shape == (k, n)
    assert y_nn.shape == (k, m)
    assert weights.shape == (k,)

    N_coefs = n_coef(n, delta)

    assert k >= 2 * N_coefs

    W = np.diagflat(weights)

    Z = np.empty((k, N_coefs), dtype=float)
    for i in range(k):
        Z[i] = all_monomials(x_nn[i, :] - x_0, delta)

    mat = np.linalg.inv(Z.T @ W @ Z) @ Z.T @ W
    assert mat.shape == (N_coefs, k)

    out = np.empty(m)

    for j in range(m):
        pj = mat @ y_nn[:, j]
        out[j] = pj[0]

    return out


def n_coef(n: int, delta: int) -> int:
    """
    Number of monomials in n variables, up to degree delta

    n : int
        Number of variables

    delta : int
        Maximum degree
    """

    out = 0
    for i in range(0, delta + 1):
        out += special.comb(n + i - 1, i)

    return int(out)


def all_monomials(x: np.ndarray, delta: int) -> np.ndarray:
    """
    Construct vector of monomials constructed from x

    Parameters
    ----------

    x : np.ndarray
        Point at which to evaluate monomial

    delta : int
        Maximum degree of monomial
    """

    n = x.shape[0]

    N_coefs = n_coef(n, delta)

    monomial_list = []

    monomial_list.append(np.ones(1))
    for deg in range(1, delta + 1):
        monomial_list.append(monomials_deg(x, deg))

    out = np.concatenate(monomial_list)
    assert out.shape == (N_coefs,)

    return out


def monomials_deg(x: np.ndarray, deg: int) -> np.ndarray:
    """
    Construct all monomials of degree equal to deg

    This brute force implementation is minimally efficient.

    Parameters
    ----------

    x : np.ndarray
        Vector from which monomials should be constructed

    deg : int
        Degree of output monomials
    """

    n = x.shape[0]

    n_terms = int(special.comb(n + deg - 1, deg))

    exponents_list = []

    # Iterate over all sets of exponents in a larger set than needed and pick
    # out the right ones
    # This is deterministic
    for e in product(range(0, deg + 1), repeat=n):
        if np.array(e).sum() == deg:
            exponents_list.append(np.array(e))

    exponents = np.array(exponents_list)
    assert exponents.shape == (n_terms, n)

    out = np.power(x, exponents).prod(axis=1)

    return out
