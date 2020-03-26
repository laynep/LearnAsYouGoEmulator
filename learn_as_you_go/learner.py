#!/bin/python

from __future__ import print_function

from typing import Callable, List, Tuple, Union

import numpy as np  # type: ignore

from .emulator import BaseEmulator
from .util import check_good


def emulate(emulator_class: BaseEmulator) -> Callable:
    """
    Emulate a function with the given emulator class

    This function is intended to be used as a decorator on non-trivial
    functions.
    The function should take one numpy array as an argument and return one
    numpy array.

    Examples
    --------

    TODO: More thorough example

    >>> from learn_as_you_go import emulate, CholeskyNnEmulator
    >>> @emulate(CholeskyNnEmulator)
    >>> def expensive_calculation(x: np.ndarray) -> np.ndarray:
    >>>     # Function definition
    >>>     return np.ones_like(x)

    """

    def inner(true_func: Callable) -> Learner:

        return Learner(true_func, emulator_class)

    return inner


class Learner(object):
    """
    A class that contains logic for learning as you go

    This class does not contain any emulation but should be constructed with an
    emulator containing emulation logic.
    The emulator must be a subclass of BaseEmulator, implementing two methods,
    `set_emul_func` and `set_emul_error_func`, that set the respective
    functions.
    """

    def __init__(self, true_func: Callable[[np.ndarray], np.ndarray], emulator_class):
        """
        Constructor for Learner class

        Parameters
        ----------
        true_func : Callable
            Function to be emulated

        emulator_class : BaseEmulator
            The emulator class to be used
        """

        self.emulator_class = emulator_class
        assert issubclass(self.emulator_class, BaseEmulator)

        self.emulator = self.emulator_class()

        self.true_func: Callable[[np.ndarray], np.ndarray] = true_func

        # TODO: Allow user to set error tolerances
        self.frac_err_local: float = 1.0
        self.abs_err_local: float = 0.05

        self.output_err: bool = False

        self.trained: bool = False
        self.num_times_trained: int = 0

        self.batchTrainX: List[np.ndarray] = []
        self.batchTrainY: List[np.ndarray] = []

        self.initTrainThresh: int = 1000
        self.otherTrainThresh: int = 5000

        self.nexact: int = 0
        self.nemul: int = 0

    def overrideDefaults(self, initTrainThresh, otherTrainThresh):
        """
        Override some of the defaults that are otherwise set in the constructor

        Parameters
        ----------
        initTrainThresh : int
            Number of data points to accumulate before first training the
            emulator

        otherTrainThresh : int
            Number of new data points to accumulate retraining the emulator
        """

        self.initTrainThresh = initTrainThresh
        self.otherTrainThresh = otherTrainThresh

    def eval_true_func(self, x: np.ndarray) -> np.ndarray:
        """
        Wrapper for evaluating true function

        You want this so that you can do some pre-processing, training, or
        saving each time the emulator gets called.

        Parameters
        ----------

        x : np.ndarray
            Independent variable.
            Assumed to be a set of vectors in :math:`R^n`.
        """

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
        self.emulator.set_emul_func(xtrain, ytrain)

        # Set the emulator function by calling to the subclass's particular method
        CV_y_err = CV_y - np.array([self.emulator.emul_func(x) for x in CV_x])[0]
        assert CV_y.shape == CV_y_err.shape  # Bizarre bugs if this isn't true
        self.emulator.set_emul_error_func(CV_x, CV_y_err)

        self.num_times_trained += 1

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

        # Separate into training and cross-validation sets
        num_cv = int(frac_cv * xdata.shape[0])

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

            self.emulator.emul_func.xtrain = np.append(
                self.emulator.emul_func.xtrain, self.batchTrainX, axis=0
            )
            self.emulator.emul_func.ytrain = np.append(
                self.emulator.emul_func.ytrain, self.batchTrainY, axis=0
            )

            self.train(self.emulator.emul_func.xtrain, self.emulator.emul_func.ytrain)

            # Empty the batch
            self.batchTrainX = []
            self.batchTrainY = []

        # Calculate a value and error to return
        val: np.ndarray
        err: float

        if self.trained:
            val, err = self.emulator.emul_func(x), self.emulator.emul_error(x)

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
