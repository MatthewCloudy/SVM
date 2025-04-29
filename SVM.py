import numpy as np
from cvxopt import matrix, solvers

class SVM:
    def __init__(self, kernel, gamma, C, coeff0=None, deg=None):
        self.alphas = None
        self.b = None
        self.K = None
        self.w = None
        self.b_alf = None
        self.X = None
        self.Y = None
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.coeff0 = coeff0
        self.deg = deg

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        self.X = X
        self.Y = y
        if self.kernel == 'rbf':
            K = np.array([[np.exp(-self.gamma*np.sum((X[i] - X[j])**2))
                       for j in range(X.shape[0])]
                          for i in range(X.shape[0])])
        else:
            K = np.array([[((self.gamma*np.dot(X[i],X[j])+self.coeff0)**self.deg)
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
        t = (self.alphas*y*X.T).T
        self.w = np.zeros(X.shape[1])
        for i in range(X.shape[0]):
            self.w += t[i]

        b_candidates = []
        for i in range(len(self.alphas)):
            if self.alphas[i] > 0:
                b_candidates.append(y[i]-np.dot(self.w, X[i]))
        self.b = np.mean(b_candidates)


    def predict(self, X_test):
        X_test = np.array(X_test)
        value = -self.b*np.ones(X_test.shape[0])
        for i in range(X_test.shape[0]):
            for j in range(self.X.shape[0]):
                if self.kernel == 'rbf':
                    value[i] += (self.alphas[j]*self.Y[j]*
                                 np.exp(-self.gamma*np.sum((self.X[j]-X_test[i])**2)))
                else:
                    value[i] += (self.alphas[j] * self.Y[j] *
                                 (self.gamma * np.dot(self.X[j], X_test[i])+self.coeff0) ** self.deg)
        return np.sign(value)