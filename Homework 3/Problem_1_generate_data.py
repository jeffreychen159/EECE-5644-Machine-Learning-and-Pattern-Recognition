import numpy as np

def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [0.25, 0.25, 0.25, 0.25] 
    
    gmmParameters['meanVectors'] = np.array([[1, 3, 2, 0], 
                                             [2, 3, 0, 2], 
                                             [1, 3, 3, 2]])
    
    gmmParameters['covMatrices'] = np.zeros((3, 3, 4))
    gmmParameters['covMatrices'][:,:,0] = np.array([[1, 0.5, 0], 
                                                    [0.5, 1, 0.3], 
                                                    [0, 0.3, 1]])
    
    gmmParameters['covMatrices'][:,:,1] = np.array([[1, -0.3, 0], 
                                                   [-0.3, 1, 0.4], 
                                                   [0, 0.4, 1]])
    
    gmmParameters['covMatrices'][:,:,2] = np.array([[1, 0, 0.2], 
                                                    [0, 1, -0.2], 
                                                    [0.2, -0.2, 1]])
    
    gmmParameters['covMatrices'][:,:,3] = np.array([[1, 0.4, -0.3], 
                                                    [0.4, 1, 0.2], 
                                                    [-0.3, 0.2, 1]])
    
    x,labels = generateDataFromGMM(N,gmmParameters)
    return x, labels

def generateDataFromGMM(N,gmmParameters):
#    Generates N vector samples from the specified mixture of Gaussians
#    Returns samples and their component labels
#    Data dimensionality is determined by the size of mu/Sigma parameters
    priors = gmmParameters['priors'] # priors should be a row vector
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0] # Data dimensionality
    C = len(priors) # Number of components
    x = np.zeros((n,N))
    labels = np.zeros((1,N))
    # Decide randomly which samples will come from each component
    u = np.random.random((1,N))
    thresholds = np.zeros((1,C+1))
    thresholds[:,0:C] = np.cumsum(priors)
    thresholds[:,C] = 1
    for l in range(C):
        indl = np.where(u <= float(thresholds[:,l]))
        Nl = len(indl[1])
        labels[indl] = (l+1)*1
        u[indl] = 1.1
        x[:,indl[1]] = np.transpose(np.random.multivariate_normal(meanVectors[:,l], covMatrices[:,:,l], Nl))
        
    return x,labels - 1


training_100_x, training_100_labels = generateData(100)
training_100 = np.vstack((training_100_x, training_100_labels)).T
np.savetxt('training_100.csv', training_100, delimiter=',')

training_500_x, training_500_labels = generateData(500)
training_500 = np.vstack((training_500_x, training_500_labels)).T
np.savetxt('training_500.csv', training_500, delimiter=',')

training_1000_x, training_1000_labels = generateData(1000)
training_1000 = np.vstack((training_1000_x, training_1000_labels)).T
np.savetxt('training_1000.csv', training_1000, delimiter=',')
        
training_5000_x, training_5000_labels = generateData(5000)
training_5000 = np.vstack((training_5000_x, training_5000_labels)).T
np.savetxt('training_5000.csv', training_5000, delimiter=',')

training_10000_x, training_10000_labels = generateData(10000)
training_10000 = np.vstack((training_10000_x, training_10000_labels)).T
np.savetxt('training_10000.csv', training_10000, delimiter=',')

test_100000_x, test_100000_labels = generateData(100000)
test_100000 = np.vstack((test_100000_x, test_100000_labels)).T
np.savetxt('test_100000.csv', test_100000, delimiter=',')
