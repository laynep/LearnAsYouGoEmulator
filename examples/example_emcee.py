"""
An example use of the `learn_as_you_go` package
"""

import numpy as np  # type: ignore

# TODO: remove NOQA when isort is fixed
from learn_as_you_go import CholeskyNnEmulator as Emulator  # NOQA
from learn_as_you_go import emulate  # NOQA


def main():

    ndim = 2

    nwalkers = 20
    niterations = 1000
    nthreads = 1

    ######################
    ######################
    # Toy likelihood
    @emulate(Emulator)
    def loglike(x):
        if x.ndim != 1:
            loglist = []
            for x0 in x:
                loglist.append(-np.dot(x0, x0))
            return np.array(loglist)
        else:
            return np.array(-np.dot(x, x))

    ######################
    ######################

    # Make fake data
    def get_x(ndim):

        if ndim == 1:

            return np.random.randn(1000)

        elif ndim == 2:

            return np.array([np.random.normal(0.0, 1.0), np.random.normal(0.0, 0.1)])
            # np.random.normal(1.0,0.1),
            # np.random.normal(0.0,0.1),
            # np.random.normal(0.0,60.1),
            # np.random.normal(1.0,2.1)])

        else:
            raise RuntimeError(
                "This number of dimensions has not been implemented for testing yet."
            )

    # Let's see if this works with a Monte Carlo method
    import emcee  # type: ignore

    p0 = np.array([get_x(ndim) for _ in range(nwalkers)])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, threads=nthreads)

    for result in sampler.sample(p0, iterations=niterations, storechain=False):
        fname = open("test.txt", "a")

        for elmn in zip(result[1], result[0]):
            fname.write("%s " % str(elmn[0]))
            for k in list(elmn[1]):
                fname.write("%s " % str(k))
            fname.write("\n")

    print("n exact evals:", loglike.nexact)
    print("n emul evals:", loglike.nemul)


def test_main():
    main()


if __name__ == "__main__":
    main()
