"""
An example use of the `learn_as_you_go` package
"""

import numpy as np

from learn_as_you_go.emulator import emulator


def main():

    ndim = 2

    ######################
    ######################
    # Toy likelihood
    @emulator
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

    if ndim == 1:
        Xtrain = get_x(ndim)
        xlist = np.linspace(-3.0, 3.0, 11)

    elif ndim == 2:

        Xtrain = np.array([get_x(ndim) for _ in range(10000)])
        xlist = np.array([get_x(ndim) for _ in range(10)])

    else:
        raise RuntimeError(
            "This number of dimensions has not been implemented for testing yet."
        )

    Ytrain = np.array([loglike(X) for X in Xtrain])
    loglike.train(
        Xtrain, Ytrain, frac_err_local=0.05, abs_err_local=1e0, output_err=False
    )

    for x in xlist:
        print("x", x)
        print("val, err", loglike(x))


if __name__ == "__main__":
    main()
