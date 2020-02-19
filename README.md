# LearnAsYouGoEmulator

Python implementation of the k-nearest neighbors Monte Carlo accelerator published in http://arxiv.org/abs/arXiv:1506.01079.

The basic usage of the emulator code should be something like this:

```
@emulator
def loglike(x):
    if x.ndim!=1:
        loglist = []
        for x0 in x:
            loglist.append(-np.dot(x0,x0))
        return np.array(loglist)
    else:
        return np.array(-np.dot(x,x))
```

This decorates any Python scalar function and loglike is now an instance
of the emulator class, where the __call__(x) function acts similarly to
the loglike(x) function definition above.  However, it now has the ability to train a k--nearest neighbors emulator using loglike.train(...), which makes a separate emulation routine that will be called first anytime loglike(x) is evaluated.

We learn both the output of loglike(x) and the difference between the
emulator and the true value of loglike(x) so that we can make a
prediction for the error residuals.  We then put a cutoff on the amount
of absolute (or fractional) error that one will allow for any local
evaluation of the target function.  Any call to the emulator(x) that has
a too-large error will be discarded and the actual function loglike(x)
defined above will be evaluated instead.


## Installation

There are a small number of python dependencies.
If you use anaconda you can create an appropriate environment by running
```
conda env create --file environment.yml
```
from this directory.
