from __future__ import print_function

from typing import Callable, List, Tuple, Union

import numpy as np  # type: ignore

from .emulator import BaseEmulator
from .util import check_good


def emulate(emulator_class) -> Callable:
    """
    Emulate a function with the given emulator class

    This function is intended to be used as a decorator on non-trivial
    functions.
    The function should take one numpy array as an argument and return one
    numpy array.

    Examples
    --------

    TODO: More thorough example

    >>> from layg import emulate, CholeskyNnEmulator
    >>> @emulate(CholeskyNnEmulator)
    >>> def expensive_calculation(x: np.ndarray) -> np.ndarray:
    >>>     # Function definition
    >>>     return np.ones_like(x)

    """

    assert issubclass(emulator_class, BaseEmulator)

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

    Attributes
    ----------

    emulator_class : BaseEmulator
        The type of emulator used.

    emulator : BaseEmulator
        An instance of the class `emulator_class`.
        This is where the heavy lifting goes on.

    true_func : Callable
        The function which is emulated

    frac_err_local : float
        Maximum fractional error in emulated function.  Calls to emulation function
        that exceed this error level are evaluated exactly instead.
        Default: 1.0

    abs_err_local : float
        Maximum absolute error allowed in emulated function.  Calls to emulation function
        that exceed frac_err_local but are lower than abs_err_local are emulated, rather
        than exactly evaluated.
        FIXME: this doesn't happen
        Default: 0.05

    output_err : bool
        Whether to output an error estimate.
        Set to False if you do not want the error to be an output of the emulated function.
        Set to True if you do.
        Default: False

    trained : bool
        Whether the emulator has been trained

    used_train_x : List[np.ndarray]
    used_train_y : List[np.ndarray]
        Values from the true function that were used last time the emulator was
        trained

    batch_train_x : List[np.ndarray]
    batch_train_y : List[np.ndarray]
        Values from the true function that have not yet been used to train the
        emulator

    init_train_thresh : int
        Number of points to accumulate before training the emulator

    frac_cv : float
        Fraction of training set to use for error modelling
        The default value of 0.5 means that the prediction and the error are
        estimated off the same amount of data.
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

        self.frac_err_local: float = 1.0
        self.abs_err_local: float = 0.05

        self.output_err: bool = False

        self.used_train_x: List[np.ndarray] = []
        self.used_train_y: List[np.ndarray] = []

        self.batch_train_x: List[np.ndarray] = []
        self.batch_train_y: List[np.ndarray] = []

        self._size_last_trained = np.nan

        self.init_train_thresh: int = 1000

        # Fraction of training set to use for error modelling
        self.frac_cv = 0.5

        # Whether the emulator is trained
        self.trained: bool = False
        self._num_times_trained: int = 0

        # Number of exact or emulated evaluations
        self._nexact: int = 0
        self._nemul: int = 0

    def eval_true_func(self, x: np.ndarray) -> np.ndarray:
        """
        Wrapper for evaluating true function

        You want this so that you can do some pre-processing, training, or
        saving each time the emulator gets called.

        Train the emulator if the specified thresholds are met.

        Parameters
        ----------

        x : np.ndarray
            Independent variable.
            Assumed to be a set of vectors in :math:`R^n`.
        """

        myY: np.ndarray = self.true_func(x)

        # Add x, val to a batch list that we will hold around
        # TODO: lists should have no duplicates
        self.batch_train_x.append(x)
        self.batch_train_y.append(myY)

        # Add points to the emulator
        # No effect unless the emulator implements updating on the fly
        self.emulator.add_data(np.array([x]), np.array([myY]))

        retraining_threshold = (1 + self.frac_cv) * self._size_last_trained
        current_points = len(self.used_train_x) + len(self.batch_train_x)

        # Check if list size has increased above some threshold
        # If so, train for first time
        if not self.trained and len(self.batch_train_x) >= self.init_train_thresh:

            self.train(np.array(self.batch_train_x), np.array(self.batch_train_y))

            # Empty the batch
            self.used_train_x.extend(self.batch_train_x)
            self.used_train_y.extend(self.batch_train_y)
            self.batch_train_x = []
            self.batch_train_y = []

        # Check if list size has increased above retraining threshold
        # If so, train again
        elif self.trained and current_points >= retraining_threshold:

            new_train_x = np.array(self.used_train_x + self.batch_train_x)
            new_train_y = np.array(self.used_train_y + self.batch_train_y)

            self.train(new_train_x, new_train_y)

            # Empty the batch
            self.used_train_x.extend(self.batch_train_x)
            self.used_train_y.extend(self.batch_train_y)
            self.batch_train_x = []
            self.batch_train_y = []

        return myY

    def train(self, x_train: np.ndarray, y_train: np.ndarray):
        """Train a ML algorithm to replace true_func: X --> Y.  Estimate error model via cross-validation.

        Parameters
        ----------
        x_train : ndarray
            Independent variable of training set.  Assumed to be a set of
            vectors in :math:`R^n`

        y_train : ndarray
            Dependent variable of training set.  Assumed to be a set of vectors
            in :math:`R^m`, although it has limited functionality if m!=1.
        """

        print("RETRAINING!------------------------")

        # TODO: remove this
        if self.output_err is not False:
            # raise Exception('Do not currently have capability to output the error to the chain.')
            pass

        self._size_last_trained = x_train.shape[0]

        # Separate into training and cross-validation sets with 50-50 split so that
        # the prediction and the error are estimated off the same amount of data
        x_train, y_train, x_cv, y_cv = self.split_CV(x_train, y_train, self.frac_cv)

        # Set the emulator function by calling to the subclass's particular method
        self.emulator.set_emul_func(x_train, y_train)

        # Set the emulator function by calling to the subclass's particular method
        CV_y_err = y_cv - np.array([self.emulator.emul_func(x) for x in x_cv])[0]
        assert y_cv.shape == CV_y_err.shape  # Bizarre bugs if this isn't true
        self.emulator.set_emul_error_func(x_cv, CV_y_err)

        # Provide the cross validation values to the emulator
        self.emulator.add_data(x_cv, y_cv)

        self.trained = True
        self._num_times_trained += 1

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

    def emulation_is_valid(self, val: np.ndarray, err: float) -> bool:
        """
        Check if an emulated value is valid and likely accurate

        Check whether the val and error are valid values and whether the
        estimated error is acceptable.

        Parameters
        ----------

        val : np.ndarray
            Proposed value

        err : float
            Error estimate

        Returns
        -------

        bool
        """

        # Value and error should not be inf, nan, etc
        goodval: bool = check_good(val)
        gooderr: bool = check_good(err)

        # These error checks are ok even if gooderr is False; that case is caught later
        small_abs_err: bool = np.all(np.abs(err) < self.abs_err_local)
        small_frac_err: bool = np.all(np.abs(err / val) < self.frac_err_local)

        return goodval and gooderr and small_abs_err and small_frac_err

    def __call__(self, x: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
        """
        The method that is executed when the wrapped function is called

        Try to use an emulated version of `true_func`.  If this is not possible
        because the error estimate is too high or the emulator is not trained,
        fall back to the true function.

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

        val: np.ndarray
        err: float

        if self.trained:
            # Note: in the paper, it is assumed that error checking can be
            # performed before the evaluation of the emulator.  We can't do
            # this as the error check uses value for the fractional error.
            val, err = self.emulator.emul_func(x), self.emulator.emul_error(x)

            if self.emulation_is_valid(val, err):
                # print("Emulated -------", val, err#, self.true_func(x))
                self._nemul += 1
            else:
                #    print("Exact evaluation -----------",goodval,gooderr)
                val = self.eval_true_func(x)
                err = 0.0
                self._nexact += 1

        else:
            val, err = self.eval_true_func(x), 0.0
            self._nexact += 1

        if self.output_err:
            return val, float(err)
        else:
            return val
