LearnAsYouGoEmulator
====================

.. image:: https://github.com/musoke/LearnAsYouGoEmulator/workflows/pytest/badge.svg

.. image:: https://github.com/musoke/LearnAsYouGoEmulator/workflows/doc/badge.svg

.. image:: https://github.com/musoke/LearnAsYouGoEmulator/workflows/lints/badge.svg


Python implementation of the Learn As You Go algorithm published in http://arxiv.org/abs/arXiv:1506.01079.

The package defines decorators that can be applied to functions to learn their output as you go and emulate
Two emulators are included: the `k`-nearest neighbors Monte Carlo accelerator described there, and a simple neural network.

The basic usage of the emulator code is something like this::

    @emulate(CholeskyNnEmulator)
    def loglike(x):
        if x.ndim!=1:
            loglist = []
            for x0 in x:
                loglist.append(-np.dot(x0,x0))
            return np.array(loglist)
        else:
            return np.array(-np.dot(x,x))

This decorates the function ``loglike`` so that it is an instance of the `Learner` class.
The ``__call__(x)`` method acts similarly to the ``loglike(x)`` function definition above.
However, it now has the ability to train a ``k``-nearest neighbors emulator using ``loglike.train(...)``, which makes a separate emulation routine that will be called first anytime ``loglike(x)`` is evaluated.

We learn both the output of ``loglike(x)`` and the difference between the emulator and the true value of ``loglike(x)`` so that we can make a prediction for the error residuals.
We then put a cutoff on the amount of absolute (or fractional) error that one will allow for any local evaluation of the target function.
Any call to the emulator(x) that has a too-large error will be discarded and the actual function ``loglike(x)`` defined above will be evaluated instead.

You can define you own emulator.
Define a class that inherits from ``BaseEmulator`` and define two methods on it: ``set_emul_func(self, x_train: np.ndarray, y_train: np.ndarray)`` and ``set_emul_error_func(self, x_train: np.ndarray, y_train: np.ndarray)`` that set values for, respectively, ``self.emul_func`` and ``self.emul_error_func``.
These values should be callable.
The logic for choosing whether to use a value from the true function or the emulated function are contained in the ``Learner`` class.
An example of this definition is provided.


Installation
------------

There are a small number of python dependencies.
If you use anaconda you can create an appropriate environment and install to your python path by running ::

    conda env create --file environment.yml
    pip install -e .

from this directory.

The ``pytorch`` dependency is only needed if you are using the neural network emulator or running the associated tests.
