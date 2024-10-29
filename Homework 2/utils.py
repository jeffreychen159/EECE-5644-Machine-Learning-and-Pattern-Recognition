import numpy as np

def evalGaussian(x, mu, Sigma):
    """
    Evaluates the Gaussian PDF N(mu, Sigma) at each column of x.
    
    Parameters:
    x (ndarray): n x N array, where each column represents a data point.
    mu (ndarray): n x 1 array representing the mean vector.
    Sigma (ndarray): n x n array representing the covariance matrix.
    
    Returns:
    ndarray: 1 x N array of evaluated Gaussian PDF values.
    """
    
    n, N = x.shape
    C = ((2 * np.pi) ** n * np.linalg.det(Sigma)) ** (-0.5)
    diff = x - mu.reshape(-1, 1)
    E = -0.5 * np.sum(diff * (np.linalg.inv(Sigma) @ diff), axis=0)
    g = C * np.exp(E)
    return g
    