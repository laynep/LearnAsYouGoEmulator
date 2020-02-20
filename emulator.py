#!/bin/python

from __future__ import print_function

import numpy as np
import scipy.interpolate as interp
from scipy.spatial import cKDTree as KDTree
import scipy.optimize as opt
import math

import sys

import matplotlib.pyplot as plt


def test_good(x):
    """Tests if scalar is infinity, NaN, or None.

    Parameters
    ----------
    x : scalar
        Input to test.

    Results
    -------
    good : logical
        False if x is inf, NaN, or None; True otherwise."""

    good = False

    #DEBUG
    return True

    if x.ndim==0:

        if x==np.inf or x==-np.inf or x is None or math.isnan(x):
            good = False
        else:
            good = True

    else:
        x0 = x.flatten()
        if any(x0==np.inf) or any(x==-np.inf) or any(x is None) or math.isnan(x0):
            good = False
        else:
            good = True

    return good

class regressor(object):
    """Basic regression techniques."""

    def split_CV(self,xdata,ydata,frac_cv):
        """Splits a dataset into a cross-validation and training set.  Shuffles the data.

        Parameters
        ----------
        xdata : ndarray
            Independent variable of dataset.  Assumed to be a set of vectors in R^n

        ydata : ndarray
            Dependent variable of dataset.  Assumed to be a set of vectors in R^m.

        frac_cv : scalar
            Fraction of dataset to be put into the cross-validation set.

        Results
        -------
        xtrain : ndarray
            Independent variable of training set.  Assumed to be a set of vectors in R^n

        ytrain : ndarray
            Dependent variable of training set.  Assumed to be a set of vectors in R^m.

        x_cv : ndarray
            Independent variable of cross-validation set.  Assumed to be a set of vectors in R^n

        y_cv : ndarray
            Dependent variable of cross-validation set.  Assumed to be a set of vectors in R^m.
        """

        #Separate into training and cross-validation sets with 80-20 split
        num_cv = int(frac_cv*xdata.shape[0])
        num_train = xdata.shape[0]-num_cv

        #Pre-process data
        #mean_vals = np.array([np.mean(col) for col in xdata.T])
        #rms_vals = np.array([np.sqrt(np.mean(col**2)) for col in xdata.T])

        rand_subset=np.arange(xdata.shape[0])
        np.random.shuffle(rand_subset)

        xdata=np.array([xdata[rand_index] for rand_index in rand_subset])
        ydata=np.array([ydata[rand_index] for rand_index in rand_subset])

        x_cv = xdata[-num_cv:]
        y_cv = ydata[-num_cv:]

        xtrain = xdata[0:-num_cv]
        ytrain = ydata[0:-num_cv]

        return xtrain, ytrain, x_cv, y_cv

    def interpolator(self,xdata,ydata):

        if xdata.shape[0]!=ydata.shape[0]:
            raise TypeError('The x and y data do not have the same number of elements.')


        ndimx = len(xdata.shape)
        ndimy = len(ydata.shape)

        if ndimy>1:
            raise TypeError('Cannot interpolate when range has higher dimensions than 1.')

        if ndimx>1:
            raise TypeError('The interpolator is not yet set up for higher dimensions than ',ndim-1)


        interp_funct = interp.interp1d(xdata,ydata)
        xmin = np.min(xdata)
        xmax = np.max(xdata)

        @np.vectorize
        def predict(x):
            if x<xmin or x>xmax:
                pred = np.inf
            else:
                pred = interp_funct(x)
            return pred

        return predict

    class cholesky_NN(object):

        def __init__(self,xdata,ydata):

            #Do some tests here

            #Find data covariance
            cov = np.cov(xdata.T)

            #Cholesky decompose to make new basis
            L_mat = np.linalg.cholesky(cov)
            self.L_mat = np.linalg.inv(L_mat)

            #Transform xdata into new basis
            self.xtrain = xdata
            self.transf_x = np.array([np.dot(self.L_mat,x) for x in xdata])

            #DEBUG
            #plt.plot(xdata[:,0],xdata[:,1],'.',color='r')
            #plt.plot(self.transf_x[:,0],self.transf_x[:,1],'.')
            #plt.show()
            #sys.exit()

            #Store training
            self.ytrain = ydata

            #Build KDTree for quick lookup
            self.transf_xtree = KDTree(self.transf_x)

        def __call__(self,x,k=5):

            if k<2:
                raise Exception("Need k>1")
            if x.ndim != self.xtrain[0].ndim:
                raise Exception("Requested x and training set do not have the same number of dimension.")

            #Change basis
            x0 = np.dot(self.L_mat,x)

            #Get nearest neighbors
            dist, loc = self.transf_xtree.query(x0,k=k)
            #Protect div by zero
            dist = np.array([np.max([1e-15,d]) for d in dist])
            weight = 1.0/dist
            nearest_y = self.ytrain[loc]

            #Interpolate with weighted average
            if self.ytrain.ndim > 1:
                y_predict = np.array([np.average(y0,weights=weight) for y0 in nearest_y.T])
                testgood = all([test_good(y) for y in y_predict])
            elif self.ytrain.ndim==1:
                y_predict = np.average(nearest_y,weights=weight)
                testgood = test_good(y_predict)
            else:
                raise Exception('The dimension of y training data is weird')


            if not testgood:
                raise Exception('y prediction went wrong')

            return y_predict


        def train_dist_error_model(self,xtrain,ytrain,k=5):
            """Rather than learning a non-parametric error model, we can define a parametric error model instead and learn its parameters."""

            if xtrain.shape[0]!=ytrain.shape[0]:
                raise TypeError('Xtrain and Ytrain do not have same shape.')

            dist_list = []
            for x0 in xtrain:

                #Change basis
                x0 = np.dot(self.L_mat,x0)

                #Get nearest neighbors in original training set
                dist, loc = self.transf_xtree.query(x0,k=k)
                #Weighted density in ball for NN
                #dist = np.array([np.max([1e-15,d]) for d in dist])
                #weight = 1.0/dist
                #dist_list.append(np.sum(weight))
                dist_list.append(np.mean(dist))

            dist_list = np.array(dist_list)

            def error_model(dist, a, b, c):
                return a*(dist) + b*(dist)**c

            bestfit, cov = opt.curve_fit(error_model,
                    dist_list,np.abs(ytrain),
                    #bounds=((0.0,0.0,0.0),(np.inf,np.inf,np.inf)))
                    bounds=((0.0,0.0,0.0),(1e1,1e1,1e1)))

            #print("this is bestfit:", bestfit)

            def new_error_model(xval):
                xval = np.dot(self.L_mat,xval)
                #Get nearest neighbors in original training set
                dist, loc = self.transf_xtree.query(xval,k=k)
                #Mean distance to NN
                dist = np.mean(dist)

                #dist = dist/bestfit[2]

                err_guess = bestfit[0]*dist + bestfit[1]*dist**bestfit[2]
                rand_sign = np.random.rand() - 0.5
                #err_guess *= 1.0 if rand_sign>0.0 else -1.0

                return err_guess


            #DEBUG
            #plt.plot(dist_list, np.abs(ytrain),'bo')
            #plt.plot(dist_list, map(new_error_model,xtrain),'ro')
            #plt.show()


            return new_error_model

#Emulator
class emulator(regressor):

    def __init__(self, true_func):

        self.true_func = true_func
        self.emul_func = self.true_func

        self.frac_err_local = 0.0
        self.abs_err_local = 0.0
        self.output_err = False

        self.trained = False

        self.batchTrainX = []
        self.batchTrainY = []

        self.initTrainThresh = 1000
        self.otherTrainThresh = 5000

        #DEBUG
        self.nexact = 0
        self.nemul = 0

    def overrideDefaults(self, initTrainThresh, otherTrainThresh):
        """Override some of the defaults that are otherwise set
        in the constructor."""

        self.initTrainThresh = initTrainThresh
        self.otherTrainThresh = otherTrainThresh


    def eval_true_func(self,x):
        """Wrapper for real emulating function.  You want this so that
        you can do some pre-processing, training, or saving each time
        the emulator gets called."""

        myY = self.true_func(x)

        #Add x, val to a batch list that we will hold around
        self.batchTrainX.append(x)
        self.batchTrainY.append(myY)

        return myY


    def train(self, xtrain, ytrain,frac_err_local=1.0,abs_err_local=0.05,output_err=False):
        """Train a ML algorithm to replace true_func: X --> Y.  Estimate error model via cross-validation.

        Parameters
        ----------
        xtrain : ndarray
            Independent variable of training set.  Assumed to be a set of vectors in R^n

        ytrain : ndarray
            Dependent variable of training set.  Assumed to be a set of scalars in R^m, although it has
            limited functionality if m!=1.

        frac_err_local : scalar
            Maximum fractional error in emulated function.  Calls to emulation function
            that exceed this error level are evaluated exactly instead.

        abs_err_local : scalar
            Maximum absolute error allowed in emulated function.  Calls to emulation function
            that exceed frac_err_local but are lower than abs_err_local are emulated, rather
            than exactly evaluated.

        output_err : logical
            Set to False if you do not want the error to be an output of the emulated function.
            Set to True if you do.
        """

        print("RETRAINING!------------------------")

        self.frac_err_local = frac_err_local
        self.abs_err_local = abs_err_local

        self.trained = True

        if not output_err==False:
            #raise Exception('Do not currently have capability to output the error to the chain.')
            pass

        self.output_err = output_err

        #Separate into training and cross-validation sets with 50-50 split so that
        #the prediction and the error are estimated off the same amount of data

        frac_cv = 0.5
        xtrain, ytrain, CV_x, CV_y = self.split_CV(xtrain, ytrain, frac_cv)

        self.emul_func = self.cholesky_NN(xtrain,ytrain)
        CV_y_err = CV_y - np.array([ self.emul_func(x) for x in CV_x  ])

        self.emul_error = self.emul_func.train_dist_error_model(CV_x,CV_y_err)
        self.emul_error2 = self.cholesky_NN(CV_x,CV_y_err)

        #xtest =[2.0* np.array(np.random.randn(2)) for _ in range(10)]
        #for x in xtest:
        #    print("--------------")
        #    print("x", x)
        #    print("prediction:", self.emul_func(x))
        #    print("error param:", self.emul_error(x))
        #    print("error nonparam:", self.emul_error2(x))
        #    print("real val, real err:", self.true_func(x), self.true_func(x) - self.emul_func(x))

        #sys.exit()

        #self.emul_func = self.interpolator(xtrain,ytrain)
        #CV_y_err = CV_y - self.emul_func(CV_x)
        #self.emul_error = self.interpolator(CV_x,CV_y_err)


    def __call__(self,x):

        #Check if list size has increased above some threshold
        #If so, retrain.  Else, skip it
        if (not self.trained and len(self.batchTrainX)>self.initTrainThresh) or (self.trained and len(self.batchTrainX)>self.otherTrainThresh):

            if self.trained:

                self.emul_func.xtrain = np.append(self.emul_func.xtrain, self.batchTrainX,axis=0)
                self.emul_func.ytrain = np.append(self.emul_func.ytrain, self.batchTrainY,axis=0)


                self.train(self.emul_func.xtrain,self.emul_func.ytrain)

            else:

                self.train(np.array(self.batchTrainX),np.array(self.batchTrainY))


            #Empty the batch
            self.batchTrainX = []
            self.batchTrainY = []


        if self.trained:
            val, err = self.emul_func(x), self.emul_error(x)
        else:
            val, err = self.eval_true_func(x), 0.0

        goodval = test_good(val)
        gooderr = test_good(err)

        #Absolute error has to be under threshold, then checks fractional error vs threshold
        if gooderr:
            try:
                gooderr = all(np.abs(err)<self.abs_err_local)
            except:
                gooderr = np.abs(err)<self.abs_err_local
        if gooderr:
            try:
                gooderr = all(np.abs(err/val)<self.frac_err_local)
            except:
                gooderr = np.abs(err/val)<self.frac_err_local


        #DEBUG
        if not goodval or not gooderr:
            #if self.trained:
            #    print("Exact evaluation -----------",goodval,gooderr)
            self.nexact += 1
            val = self.eval_true_func(x)
            err = 0.0
        else:
            if self.trained:
                self.nemul += 1
                #print("Emulated -------", val, err#, self.true_func(x))

        if self.output_err:
            return float(val), float(err)
        else:
            return float(val)


def main():


    ndim = 2

    nwalkers = 20
    niterations = 1000
    nthreads = 1

    #Make fake data

    def get_x(ndim):

        if ndim==1:

            return np.random.randn(1000)

        elif ndim==2:

            return np.array([np.random.normal(0.0,1.0),
                np.random.normal(0.0,0.1)])
                #np.random.normal(1.0,0.1),
                #np.random.normal(0.0,0.1),
                #np.random.normal(0.0,60.1),
                #np.random.normal(1.0,2.1)])

        else:
            raise RuntimeError('This number of dimensions has'+
                    ' not been implemented for testing yet.')

    if ndim==1:
        Xtrain = get_x(ndim)
        xlist = np.linspace(-3.0,3.0,11)

    elif ndim==2:

        Xtrain = np.array([get_x(ndim) for _ in range(10000)])
        xlist = np.array([get_x(ndim) for _ in range(10)])

    else:
        raise RuntimeError('This number of dimensions has'+
                ' not been implemented for testing yet.')


    #Ytrain = np.array([loglike(X) for X in Xtrain])
    #loglike.train(Xtrain,Ytrain,frac_err_local=0.05,abs_err_local=1e0,output_err=True)

    ######################
    ######################
    #Toy likelihood
    @emulator
    def loglike(x):
        if x.ndim!=1:
            loglist = []
            for x0 in x:
                loglist.append(-np.dot(x0,x0))
            return np.array(loglist)
        else:
            return np.array(-np.dot(x,x))
    ######################
    ######################

    for x in xlist:
        print("x", x)
        print("val, err", loglike(x))

    #Let's see if this works with a Monte Carlo method
    import emcee

    p0 = np.array([get_x(ndim) for _ in range(nwalkers)])
    sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, threads=nthreads)

    for result in sampler.sample(p0, iterations=niterations, storechain=False):
        fname = open('test.txt', "a")

        for elmn in zip(result[1],result[0]):
            fname.write("%s " % str(elmn[0]))
            for k in list(elmn[1]):
                fname.write("%s " % str(k))
            fname.write("\n")

    print("n exact evals:", loglike.nexact)
    print("n emul evals:", loglike.nemul)


if __name__=="__main__":
    main()
