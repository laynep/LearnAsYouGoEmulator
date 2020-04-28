LearnAsYouGoEmulator
====================

Python implementation of the Learn As You Go algorithm described in `arxiv:1506:01079 <http://arxiv.org/abs/arXiv:1506.01079>`_ and `arxiv:2004.11929 <https://arxiv.org/abs/2004.11929>`_.

.. image:: https://badge.fury.io/py/layg.svg
    :target: https://badge.fury.io/py/layg

.. image:: https://zenodo.org/badge/240627897.svg
    :target: https://zenodo.org/badge/latestdoi/240627897

.. image:: https://readthedocs.org/projects/learnasyougoemulator/badge/?version=latest
    :target: https://learnasyougoemulator.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status


.. image:: https://github.com/auckland-cosmo/LearnAsYouGoEmulator/workflows/pytest/badge.svg
    :target: https://github.com/auckland-cosmo/LearnAsYouGoEmulator/actions?query=workflow%3Apytest

.. image:: https://github.com/auckland-cosmo/LearnAsYouGoEmulator/workflows/doc/badge.svg
    :target: https://github.com/auckland-cosmo/LearnAsYouGoEmulator/actions?query=workflow%3Adoc

.. image:: https://github.com/auckland-cosmo/LearnAsYouGoEmulator/workflows/lints/badge.svg
    :target: https://github.com/auckland-cosmo/LearnAsYouGoEmulator/actions?query=workflow%3Alints


The package defines a decorator that can be applied to functions to convert them to functions which learn outputs as they go and emulate the true function when expected errors are low.
Two emulators are included: the `k`-nearest neighbors Monte Carlo accelerator described there, and a simple neural network.

The basic usage of the emulator code is something like this::

    @emulate(CholeskyNnEmulator)
    def loglike(x):
        """
        Your complex and expensive function here
        """
        return -np.dot(x,x)

This decorates the function ``loglike`` so that it is an instance of the ``Learner`` class.
It can be used in the same way as the original function: just call it as ``loglike(x)``.

The ``__call__(x)`` method now hides some extra complexity: it uses the Learn As You Go emulation scheme.
It learns both the output of ``loglike(x)`` and the difference between the emulated and the true values of ``loglike(x)`` so that it can make a prediction of future the error residuals.
We then put a cutoff on the amount of error that one will allow for any local evaluation of the target function.
Any call to the emulator that has a too-large error will be discarded and the actual function ``loglike(x)`` defined above will be evaluated instead.

The logic for generating training sets and returning a value from either the true function or the emulated function are contained in the ``Learner`` class.
The ``Learner`` class relies on an emulator class to do the actual emulation.

You can define you own emulator.
Define a class that inherits from ``BaseEmulator`` and define two methods on it: ``set_emul_func(self, x_train: np.ndarray, y_train: np.ndarray)`` and ``set_emul_error_func(self, x_train: np.ndarray, y_train: np.ndarray)`` that set functions for, respectively, ``self.emul_func`` and ``self.emul_error_func``.
An example of this definition is provided in `examples/example_custom_emulator.py`.

See readthedocs.org_ for more documentation.


Installation
------------

**pip**

The package is available on pypi.org_.
Install it with ::

    pip install layg

**anaconda**

If you use anaconda you can create an appropriate environment and install to your python path by running ::

    conda env create --file environment.yml
    pip install -e .

from this directory.

.. _pypi.org: https://pypi.org/project/layg/

.. _readthedocs.org: https://learnasyougoemulator.readthedocs.io
