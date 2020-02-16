from __future__ import print_function
import matplotlib.pyplot as plt
from scipy.misc import logsumexp
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist
 
#to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Given a test datum, it returns its prediction based on locally weighted regression

    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter   
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    N_train = x_train.shape[0]
    norm_matrix = -l2(test_datum.transpose(), x_train)
    a = []
    s = logsumexp([norm_matrix[0]/(2*tau**2)])
    for j in range(N_train):    
        log = norm_matrix[0][j]/(2*tau**2) - s
        a.append(np.exp(log))
    a = np.array(a)
    d = np.diag(a)
    left = np.matmul(x_train.transpose(), np.matmul(d, x_train))
    right = np.matmul(x_train.transpose(), np.matmul(d, y_train))
    w = np.linalg.solve(left + 1e-8 * np.eye(left.shape[0]), right)
    return np.dot(w, test_datum)
    ## TODO

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    counter = 0
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    return losses

#to implement
def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
    loss = run_on_fold(x_test, y_test, x_train, y_train, taus)
    return loss
    ## TODO


if __name__ == "__main__":
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    losses = run_k_fold(x,y,taus,k=5)
    plt.plot(losses)
    print("min loss = {}".format(losses.min()))

