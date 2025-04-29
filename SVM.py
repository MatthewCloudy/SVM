import numpy as np
from cvxopt import matrix, solvers

class SVM:
    def __init__(self, kernel, gamma, C):
        self.alphas = None
        self.b = None
        self.K = None
        self.kernel = kernel
        self.gamma = gamma
        self.C = C

    def fit(self, X, y):
        if self.kernel == 'rbf':
            X = np.array(X)
            y = np.array(y)
            K = np.array([[np.exp(-self.gamma*np.sum((X[i] - X[j])**2))
                       for j in range(X.shape[0])]
                          for i in range(X.shape[0])])
            self.K = K
            Y = np.outer(y,y)
            P = Y*K
            q = -np.ones(X.shape[0])
            A = y.reshape(1,-1)
            G = np.vstack((-np.eye(X.shape[0]),np.eye(X.shape[0])))
            h = np.hstack((np.zeros(X.shape[0]),self.C*np.ones(X.shape[0])))
            h = h.reshape(-1, 1)
            b = 0
            solvers.options['show_progress'] = False
            sol = solvers.qp(matrix(P),matrix(q),matrix(G),matrix(h),matrix(A, tc='d'),matrix(b,tc='d'))
            self.alphas = np.array(sol['x']).flatten()
    def predict(self, X):
        pass