#!/bin/python

from __future__ import print_function

from typing import Callable, List, Tuple, Union

import numpy as np  # type: ignore

from .util import check_good


# Emulator
class Learner(object):
    """
    A class that contains logic for learning as you go

    It does not do any emulation but should be used to define a subclass
    containing emulation logic.
    The subclass must contain two methods, `set_emul_func` and
    `set_emul_error_func`, that set the respective functions.

    TODO: Example for docs
    """

    def __init__(self, true_func):

        self.true_func: Callable[np.ndarray, np.ndarray] = true_func
        self.emul_func = self.true_func
        self.emul_error: Callable[
            np.ndarray, float
        ] = lambda _: 0.0  # The initial function is exact

        # TODO: Allow user to set error tolerances
        self.frac_err_local: float = 1.0
        self.abs_err_local: float = 0.05

        self.output_err = False

        self.trained: bool = False
        self.num_times_trained: int = 0

        self.batchTrainX: List[np.ndarray] = []
        self.batchTrainY: List[np.ndarray] = []

        self.initTrainThresh = 1000
        self.otherTrainThresh = 5000

        # DEBUG
        self.nexact = 0
        self.nemul = 0

    def overrideDefaults(self, initTrainThresh, otherTrainThresh):
        """Override some of the defaults that are otherwise set
        in the constructor."""

        self.initTrainThresh = initTrainThresh
        self.otherTrainThresh = otherTrainThresh

    def eval_true_func(self, x: np.ndarray) -> np.ndarray:
        """Wrapper for real emulating function.  You want this so that
        you can do some pre-processing, training, or saving each time
        the emulator gets called."""

        myY: np.ndarray = self.true_func(x)

        # Add x, val to a batch list that we will hold around
        # TODO: lists should have no duplicates
        self.batchTrainX.append(x)
        self.batchTrainY.append(myY)

        return myY

    def train(
        self, xtrain, ytrain, frac_err_local=1.0, abs_err_local=0.05, output_err=False
    ):
        """Train a ML algorithm to replace true_func: X --> Y.  Estimate error model via cross-validation.

        Parameters
        ----------
        xtrain : ndarray
            Independent variable of training set.  Assumed to be a set of vectors in R^n

        ytrain : ndarray
            Dependent variable of training set.  Assumed to be a set of scalars in R^m, although it has
            limited functionality if m!=1.

        frac_err_local : scalar
            Maximum fractional error in emulated function.  Calls to emulation function
            that exceed this error level are evaluated exactly instead.

        abs_err_local : scalar
            Maximum absolute error allowed in emulated function.  Calls to emulation function
            that exceed frac_err_local but are lower than abs_err_local are emulated, rather
            than exactly evaluated.

        output_err : logical
            Set to False if you do not want the error to be an output of the emulated function.
            Set to True if you do.
        """

        print("RETRAINING!------------------------")

        self.frac_err_local = frac_err_local
        self.abs_err_local = abs_err_local

        self.trained = True

        if output_err is not False:
            # raise Exception('Do not currently have capability to output the error to the chain.')
            pass

        self.output_err = output_err

        # Separate into training and cross-validation sets with 50-50 split so that
        # the prediction and the error are estimated off the same amount of data

        frac_cv = 0.5
        xtrain, ytrain, CV_x, CV_y = self.split_CV(xtrain, ytrain, frac_cv)

        # Set the emulator function by calling to the subclass's particular method
        # TODO: fail gracefully if this is not defined.
        self.set_emul_func(xtrain, ytrain)

        # Set the emulator function by calling to the subclass's particular method
        CV_y_err = CV_y - np.array([self.emul_func(x) for x in CV_x])[0]
        assert CV_y.shape == CV_y_err.shape  # Bizarre bugs if this isn't true
        self.set_emul_error_func(CV_x, CV_y_err)

        self.num_times_trained += 1

        # self.emul_error2 = self.cholesky_NN(CV_x, CV_y_err)

        # xtest =[2.0* np.array(np.random.randn(2)) for _ in range(10)]
        # for x in xtest:
        #    print("--------------")
        #    print("x", x)
        #    print("prediction:", self.emul_func(x))
        #    print("error param:", self.emul_error(x))
        #    print("error nonparam:", self.emul_error2(x))
        #    print("real val, real err:", self.true_func(x), self.true_func(x) - self.emul_func(x))

        # import sys; sys.exit()

    def split_CV(self, xdata: np.ndarray, ydata: np.ndarray, frac_cv: float):
        """Splits a dataset into a cross-validation and training set.  Shuffles the data.

        Parameters
        ----------
        xdata : ndarray
            Independent variable of dataset.  Assumed to be a set of vectors in R^n

        ydata : ndarray
            Dependent variable of dataset.  Assumed to be a set of vectors in R^m.

        frac_cv : scalar
            Fraction of dataset to be put into the cross-validation set.

        Results
        -------
        xtrain : ndarray
            Independent variable of training set.  Assumed to be a set of vectors in R^n

        ytrain : ndarray
            Dependent variable of training set.  Assumed to be a set of vectors in R^m.

        x_cv : ndarray
            Independent variable of cross-validation set.  Assumed to be a set of vectors in R^n

        y_cv : ndarray
            Dependent variable of cross-validation set.  Assumed to be a set of vectors in R^m.
        """

        # Separate into training and cross-validation sets with 80-20 split
        num_cv = int(frac_cv * xdata.shape[0])

        # Pre-process data
        # mean_vals = np.array([np.mean(col) for col in xdata.T])
        # rms_vals = np.array([np.sqrt(np.mean(col**2)) for col in xdata.T])

        rand_subset = np.arange(xdata.shape[0])
        np.random.shuffle(rand_subset)

        xdata = np.array([xdata[rand_index] for rand_index in rand_subset])
        ydata = np.array([ydata[rand_index] for rand_index in rand_subset])

        x_cv = xdata[-num_cv:]
        y_cv = ydata[-num_cv:]

        xtrain = xdata[0:-num_cv]
        ytrain = ydata[0:-num_cv]

        return xtrain, ytrain, x_cv, y_cv

    def __call__(self, x: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        The method that is executed when the wrapped function is called

        Parameters
        ----------
        x : ndarray
            Independent variable.
            Assumed to be a set of vectors in :math:`R^n`.

            This is either passed directly to the true function or to an emulator.

        Results
        -------
        val : float
            The exact or emulated value of the true function.

        (val, err) : (np.adarray, float)
            The exact or emulated value of the true function and an error estimate.
            This version is returned is `return_err` is set to true on the class.
        """

        # Check if list size has increased above some threshold
        # If so, train for first time
        if not self.trained and len(self.batchTrainX) > self.initTrainThresh:

            self.train(np.array(self.batchTrainX), np.array(self.batchTrainY))

            # Empty the batch
            self.batchTrainX = []
            self.batchTrainY = []

        # Check if list size has increased above retraining threshold
        # If so, train again
        elif self.trained and len(self.batchTrainX) > self.otherTrainThresh:

            self.emul_func.xtrain = np.append(
                self.emul_func.xtrain, self.batchTrainX, axis=0
            )
            self.emul_func.ytrain = np.append(
                self.emul_func.ytrain, self.batchTrainY, axis=0
            )

            self.train(self.emul_func.xtrain, self.emul_func.ytrain)

            # Empty the batch
            self.batchTrainX = []
            self.batchTrainY = []

        # Calculate a value and error to return
        val: np.ndarray
        err: float

        if self.trained:
            val, err = self.emul_func(x), self.emul_error(x)

            # Value and error should not be inf, nan, etc
            goodval: bool = check_good(val)
            gooderr: bool = check_good(err)

            # These error checks are ok even if gooderr is False; that case is caught later
            small_abs_err: bool = np.all(np.abs(err) < self.abs_err_local)
            small_frac_err: bool = np.all(np.abs(err / val) < self.frac_err_local)

            if goodval and gooderr and small_abs_err and small_frac_err:
                # print("Emulated -------", val, err#, self.true_func(x))
                self.nemul += 1
            else:
                #    print("Exact evaluation -----------",goodval,gooderr)
                val = self.eval_true_func(x)
                err = 0.0
                self.nexact += 1

        else:
            val, err = self.eval_true_func(x), 0.0
            self.nexact += 1

        if self.output_err:
            return float(val), float(err)
        else:
            return float(val)
