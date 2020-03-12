import numpy as np  # type: ignore
import pytest  # type: ignore

# TODO: remove NOQA when isort is fixed
from learn_as_you_go import CholeskyNnEmulator  # NOQA


XDIM = 1
CONSTANT = 1.0
Emulator = CholeskyNnEmulator


def constant(x):
    return np.array([CONSTANT])


emulator_default_threshold = Emulator(constant)

emulator_custom_threshold = Emulator(constant)
emulator_custom_threshold.overrideDefaults(100, 1000)


def test_can_set_initial_threshold():
    """
    Test that the initial training threshold can be set
    """

    default_threshold = emulator_default_threshold.initTrainThresh
    custom_threshold = emulator_custom_threshold.initTrainThresh

    assert not default_threshold == custom_threshold


@pytest.mark.parametrize(
    "emulator", [emulator_default_threshold, emulator_custom_threshold]
)
def test_initial_training_threshold(emulator):
    """
    Test that the initial training threshold is respected

    The emulator should remain in an untrained state for the first N calls,
    then train for the first time.
    """

    training_threshold = emulator.initTrainThresh

    # The emulator should be untrained until the threshold is reached
    for i, x in enumerate(
        [np.random.uniform(size=XDIM) for _ in range(training_threshold + 2)]
    ):
        assert not emulator.trained
        assert emulator.nexact == i
        emulator(x)

    # The emulator should now be trained with one emulated sim done
    assert emulator.trained
    assert emulator.num_times_trained == 1
    assert emulator.nemul == 1
    assert emulator.nexact == training_threshold + 1


def test_retraining_threshold():
    """
    Test that retraining happens at right threshold
    """

    emulator = Emulator(constant)
    training_threshold = emulator.initTrainThresh

    # The emulator should be untrained until the threshold is reached
    for i, x in enumerate(
        [np.random.uniform(size=XDIM) for _ in range(training_threshold + 2)]
    ):
        assert not emulator.trained
        assert emulator.nexact == i
        emulator(x)

    # The emulator should now be trained with one emulated sim done
    assert emulator.trained
    assert emulator.num_times_trained == 1
    assert emulator.nemul == 1
    assert emulator.nexact == training_threshold + 1

    # The emulator should retrain when the retraining threshold is reached
    retraining_threshold = emulator.otherTrainThresh

    # Do these evaluations on an interval far from the initial interval, so
    # that they are all from the true function
    for i, x in enumerate(
        [100 + np.random.uniform(size=XDIM) for _ in range(retraining_threshold + 2)]
    ):
        emulator(x)

    # The emulator should now be trained with one emulated sim done
    assert emulator.trained
    assert emulator.num_times_trained == 2
    assert emulator.nemul == 2
    assert emulator.nexact == training_threshold + retraining_threshold + 2
