import numpy as np  # type: ignore
import pytest  # type: ignore

from learn_as_you_go.emulator import emulator


@pytest.mark.xfail
def test_default_initial_training_threshold():
    """
    Test that the default initial training threshold of 1000 is used

    The emulator should remain in an untrained state for the first 1000 calls,
    then train for the first time.
    """

    xdim = 2
    CONSTANT = 1.0
    training_threshold = 1000

    @emulator
    def constant(x):
        return CONSTANT

    for x in [np.random.uniform(size=xdim) for _ in range(training_threshold)]:
        assert not constant.trained
        constant(x)

    assert constant.trained


@pytest.mark.xfail
def test_custom_initial_training_threshold():
    """
    Test that the initial training threshold can be chosen

    The emulator should remain in an untrained state for the first N calls,
    then train for the first time.
    """

    xdim = 2
    CONSTANT = 1.0
    training_threshold = 1000
    retraining_threshold = 1000

    @emulator
    def constant(x):
        return CONSTANT

    emulator.overrideDefaults(training_threshold, retraining_threshold)

    for x in [np.random.uniform(size=xdim) for _ in range(training_threshold)]:
        assert not constant.trained
        constant(x)

    assert constant.trained
