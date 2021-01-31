import numpy as np
from numpy.linalg import solve
from findMin import findMin
from scipy.optimize import approx_fprime
import utils

# Ordinary Least Squares
class LeastSquares:
    def fit(self,X,y):
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        return X@self.w

# Least squares where each sample point X has a weight associated with it.
class WeightedLeastSquares(LeastSquares): # inherits the predict() function from LeastSquares
    def fit(self,X,y,z):
        X = z@X
        self.w = solve(X.T@X, X.T@y)

class LinearModelGradient(LeastSquares):

    def fit(self,X,y):
        n, d = X.shape

        # Initial guess
        self.w = np.zeros((d, 1))

        # check the gradient
        estimated_gradient = approx_fprime(self.w, lambda w: self.funObj(w,X,y)[0], epsilon=1e-6)
        implemented_gradient = self.funObj(self.w,X,y)[1]
        if np.max(np.abs(estimated_gradient - implemented_gradient) > 1e-4):
            print('User and numerical derivatives differ: %s vs. %s' % (estimated_gradient, implemented_gradient));
        else:
            print('User and numerical derivatives agree.')

        self.w, f = findMin(self.funObj, self.w, 100, X, y)

    def funObj(self,w,X,y):
        N, D = X.shape
        # Calculate the function value
        f = np.sum(np.log(np.exp(X@w - y) + np.exp(y - X@w)))
        # Calculate the gradient value
        Q = np.zeros(N)
        for i in range(N):
            a = np.exp(w.T@X[i] - y[i])
            b = np.exp(y[i] - w.T@X[i])
            Q[i] = (a - b) / (a + b)
        g = X.T@Q
        return (f, g)


# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        N,D = X.shape
        ones = np.ones(N)
        X = np.column_stack((ones, X))
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        N,D = X.shape
        ones = np.ones(N)
        X = np.column_stack((ones, X))
        return X@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.leastSquares = LeastSquares()
        self.p = p

    def fit(self,X,y):
        Xi = self.__polyBasis(X)
        self.w = solve(Xi.T@Xi, Xi.T@y)

    def predict(self, X):
        Xi = self.__polyBasis(X)
        return Xi@self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        N,D = X.shape
        Z = np.ones((N, self.p + 1))
        for i in range(N):
            for p in range(Z.shape[1]):
                Z[i][p] = X[i]**p
        return Z
