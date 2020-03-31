Custom Emulators
================

This example shows how to build a custom emulator by defining a subclass of :class:`learn_as_you_go.emulator.BaseEmulator`.

The emulator simply learns the mean and standard deviation of the supplied training data.

In this example the emulated function is very simple: it returns real numbers drawn from a Gaussian distribution with some mean.


.. plot:: ../../examples/example_custom_emulator.py
    :include-source:
