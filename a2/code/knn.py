"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the training data
        self.y = y 

    def predict(self, Xtest):
        T, D = Xtest.shape
        N, D = self.X.shape
        dists = utils.euclidean_dist_squared(self.X, Xtest)
        sortedDists = np.argsort(dists, axis=0)
        y_pred = np.empty(T)
        for ti in range(T):
            y_pred[ti] = stats.mode(self.y[sortedDists[:self.k, ti]])[0][0]
        # print(y_pred)
        return y_pred
