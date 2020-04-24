import numpy as np  # type: ignore
import pytest  # type: ignore

# TODO: remove NOQA when isort is fixed
from layg import CholeskyNnEmulator  # NOQA
from layg import Learner  # NOQA

XDIM = 1
CONSTANT = 1.0
Emulator = CholeskyNnEmulator


def constant(x):
    return np.array([CONSTANT])


emulator_default_threshold = Learner(constant, CholeskyNnEmulator)

emulator_custom_threshold = Learner(constant, CholeskyNnEmulator)
emulator_custom_threshold.init_train_thresh = 100


def test_can_set_initial_threshold():
    """
    Test that the initial training threshold can be set
    """

    default_threshold = emulator_default_threshold.init_train_thresh
    custom_threshold = emulator_custom_threshold.init_train_thresh

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

    training_threshold = emulator.init_train_thresh

    # The emulator should be untrained until the threshold is reached
    for i, x in enumerate(
        [np.random.uniform(size=XDIM) for _ in range(training_threshold)]
    ):
        assert not emulator.trained
        assert emulator._nexact == i
        emulator(x)

    emulator(np.random.uniform(size=XDIM))

    # The emulator should now be trained with one emulated sim done
    assert emulator.trained
    assert emulator._num_times_trained == 1
    assert emulator._nemul == 1
    assert emulator._nexact == training_threshold


def test_retraining_threshold():
    """
    Test that retraining happens at right threshold
    """

    emulator = Learner(constant, Emulator)
    training_threshold = emulator.init_train_thresh

    # The emulator should be untrained until the threshold is reached
    for i, x in enumerate(
        [np.random.uniform(size=XDIM) for _ in range(training_threshold)]
    ):
        assert not emulator.trained
        assert emulator._nexact == i
        emulator(x)

    emulator(np.random.uniform(size=XDIM))

    # The emulator should now be trained with one emulated sim done
    assert emulator.trained
    assert emulator._num_times_trained == 1
    assert emulator._nemul == 1
    assert emulator._nexact == training_threshold

    # The emulator should retrain when the retraining threshold is reached
    retraining_threshold = int((1 + emulator.frac_cv) * emulator.init_train_thresh)

    # Do these evaluations on an interval far from the initial interval, so
    # that they are all evaluated from the true function
    for i, x in enumerate(
        [
            100 + np.random.uniform(size=XDIM)
            for _ in range(retraining_threshold - training_threshold)
        ]
    ):
        assert emulator._num_times_trained == 1
        assert emulator._nexact == training_threshold + i
        emulator(x)

    emulator(np.random.uniform(size=XDIM))

    # The emulator should now be trained with one emulated sim done
    assert emulator.trained
    assert emulator._num_times_trained == 2
    assert emulator._nemul == 2
    assert emulator._nexact == retraining_threshold
