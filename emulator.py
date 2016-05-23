#!/bin/python
import numpy as np
import emcee
import scipy.interpolate as interp
from scipy.spatial import KDTree as KDTree
import scipy.optimize as opt
import math

import sys


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
        mean_vals = np.array([np.mean(col) for col in xdata.T])
        rms_vals = np.array([np.sqrt(np.mean(col**2)) for col in xdata.T])
        print "these are rms_vals", rms_vals
        print "these are mean_vals", mean_vals

        print "do data preprocessing by writing function"
        sys.exit()


        test=np.arange(xdata.shape[0])
        np.random.shuffle(test)

        xdata=np.array([xdata[tester] for tester in test])
        ydata=np.array([ydata[tester] for tester in test])

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
            self.L_mat = L_mat

            #Transform xdata into new basis
            self.xtrain = xdata
            self.transf_x = np.array([np.dot(L_mat,x) for x in xdata])

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

            def error_model(dist, a, b, c, d):
                return a*(dist/c) + b*(dist/c)**d

            bestfit, cov = opt.curve_fit(error_model,
                    dist_list,np.abs(ytrain))

            print "this is bestfit:", bestfit

            def new_error_model(xval):
                xval = np.dot(self.L_mat,xval)
                #Get nearest neighbors in original training set
                dist, loc = self.transf_xtree.query(xval,k=k)
                #Mean distance to NN
                dist = np.mean(dist)

                dist = dist/bestfit[2]

                err_guess = bestfit[0]*dist + bestfit[1]*dist**bestfit[3]
                rand_sign = np.random.rand() - 0.5
                #err_guess *= 1.0 if rand_sign>0.0 else -1.0

                return err_guess


            import matplotlib.pyplot as plt
            plt.plot(dist_list, np.abs(ytrain),'bo')
            plt.plot(dist_list, map(new_error_model,xtrain),'ro')
            plt.show()


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


    def train(self, xtrain, ytrain,frac_err_local=0.01,abs_err_local=0.0,output_err=False):
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

        self.frac_err_local = frac_err_local
        self.abs_err_local = abs_err_local

        self.trained = True

        if not output_err==False:
            #raise Exception('Do not currently have capability to output the error to the chain.')
            pass

        self.output_err = output_err

        #Separate into training and cross-validation sets with 50-50 split so that
        #the prediction and the error are estimated off the same amount of data
        frac_cv = 0.15
        xtrain, ytrain, CV_x, CV_y = self.split_CV(xtrain, ytrain, frac_cv)

        self.emul_func = self.cholesky_NN(xtrain,ytrain)
        CV_y_err = CV_y - np.array([ self.emul_func(x) for x in CV_x  ])

        self.emul_error = self.emul_func.train_dist_error_model(CV_x,CV_y_err)
        self.emul_error2 = self.cholesky_NN(CV_x,CV_y_err)

        #xtest =[2.0* np.array(np.random.randn(2)) for _ in xrange(10)]
        #for x in xtest:
        #    print "--------------"
        #    print "x", x
        #    print "prediction:", self.emul_func(x)
        #    print "error param:", self.emul_error(x)
        #    print "error nonparam:", self.emul_error2(x)
        #    print "real val, real err:", self.true_func(x), self.true_func(x) - self.emul_func(x)

        #sys.exit()


        #self.emul_func = self.interpolator(xtrain,ytrain)
        #CV_y_err = CV_y - self.emul_func(CV_x)
        #self.emul_error = self.interpolator(CV_x,CV_y_err)



    def __call__(self,x):

        if self.trained:
            val, err = self.emul_func(x), self.emul_error(x)
        else:
            val, err = self.true_func(x), 0.0

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


        if not goodval or not gooderr:
            if self.trained:
                print "Exact evaluation -----------",goodval,gooderr
            val = self.true_func(x)
            err = 0.0
        else:
            print "Emulated -------"

        if self.output_err:
            return float(val), float(err)
        else:
            return float(val)


def main():

    #Fake likelihood
    @emulator
    def loglike(x):
        if x.ndim!=1:
            loglist = []
            for x0 in x:
                loglist.append(-np.dot(x0,x0))
            return np.array(loglist)
        else:
            return np.array(-np.dot(x,x))

    ndim = 2

    if ndim==1:
        Xtrain = np.random.randn(1000)
        xlist = np.linspace(-3.0,3.0,11)

    elif ndim==2:

        def get_x():
            return np.array([np.random.normal(0.0,1.0),
                np.random.normal(0.0,0.1),
                np.random.normal(1.0,0.1),
                np.random.normal(0.0,0.1),
                np.random.normal(0.0,60.1),
                np.random.normal(1.0,2.1)])

        Xtrain = np.array([get_x() for _ in xrange(10000)])
        xlist = np.array([get_x() for _ in xrange(10)])

    Ytrain = np.array([loglike(X) for X in Xtrain])

    loglike.train(Xtrain,Ytrain,frac_err_local=0.05,abs_err_local=5e-1,output_err=True)

    for x in xlist:
        print "x", x
        print "val, err", loglike(x)



if __name__=="__main__":
    main()
