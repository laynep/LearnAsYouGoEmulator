"""
An example use of the `learn_as_you_go` package with emcee
"""

import emcee  # type: ignore
import gif  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

# TODO: remove NOQA when isort is fixed
from layg import CholeskyNnEmulator  # NOQA
from layg import emulate  # NOQA


def main():

    ndim = 2

    nwalkers = 20
    niterations = 1000
    nthreads = 1

    np.random.seed(1234)

    # Toy likelihood
    @emulate(CholeskyNnEmulator)
    def loglike(x):
        return np.array([-np.dot(x, x) ** 1])

    loglike.output_err = True
    loglike.abs_err_local = 2

    # Starting points for walkers
    p0 = np.random.normal(-1.0, 1.0, size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, threads=nthreads)

    # Sample with emcee
    with open("test.txt", "w") as f:
        for result in sampler.sample(p0, iterations=niterations, storechain=True):

            for pos, lnprob, err in zip(result[0], result[1], result[3]):
                for k in list(pos):
                    f.write("%s " % str(k))
                f.write("%s " % str(lnprob))
                f.write("%s " % str(err))
                f.write("\n")

    print("n exact evals:", loglike._nexact)
    print("n emul evals:", loglike._nemul)

    # Plot points sampled
    nframes = 50
    duration = 10
    frames = []
    lim = (-3, 3)

    for i in range(0, niterations * nwalkers, niterations * nwalkers // nframes):
        x = sampler.chain.reshape(niterations * nwalkers, ndim)[:i]
        y = np.array(sampler.blobs).reshape(niterations * nwalkers)[:i]
        frame = plot(x, y, lim)
        frames.append(frame)

    gif.save(frames, "mc.gif", duration=duration)


@gif.frame
def plot(x, err, lim):

    true = x[err == 0.0]
    emul = x[err != 0.0]

    plt.figure(figsize=(5, 5), dpi=100)

    marker = "."
    alpha = 0.3

    plt.scatter(true[:, 0], true[:, 1], marker=marker, alpha=alpha, label="true")
    plt.scatter(emul[:, 0], emul[:, 1], marker=marker, alpha=alpha, label="emulated")

    plt.xlim(lim)
    plt.ylim(lim)

    legend = plt.legend(loc=1)
    for lh in legend.legendHandles:
        lh.set_alpha(1)


def test_main():
    main()


if __name__ == "__main__":
    main()
