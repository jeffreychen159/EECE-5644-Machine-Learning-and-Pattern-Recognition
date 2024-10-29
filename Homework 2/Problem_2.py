import numpy as np
import matplotlib.pyplot as plt

def main():
    # Generate synthetic data for training and testing
    x_train, y_train, x_val, y_val = hw2q2()
    phi = generate_features(x_train)

    # Generates data for 
    fig1, axes1 = plt.subplots(2, 1, sharex=False, sharey=True)

    fig2, axes2 = plt.subplots(2, 1, sharex=False, sharey=True)

    
    # Plot training data
    for i in range(len(x_val[0])):
        axes1[0].plot(x_val[0][i], y_val[i], color='salmon', marker='*', alpha=0.4)
        axes1[1].plot(x_val[1][i], y_val[i], color='limegreen', marker='*', alpha=0.4)
        axes2[0].plot(x_val[0][i], y_val[i], color='violet', marker='*', alpha=0.4)
        axes2[1].plot(x_val[1][i], y_val[i], color='royalblue', marker='*', alpha=0.4)
    
    x_0 = np.linspace(min(x_val[0]), max(x_val[0]), num=1000)
    x_1 = np.linspace(min(x_val[1]), max(x_val[1]), num=1000)


    gammas = np.logspace(-4, 4, 9)
    for gamma in gammas:
        print("Gamma: ", + gamma)

        # Calculate and plot MLE predictions
        mle_w = mle(phi, y_train)
        print("MLE Error:", ase(mle_w, x_val, y_val))

        w = mle_w.ravel()
        axes1[0].plot(x_0, [w[0] + w[1]*x + w[3]*x**2 + w[5]*x**3 for x in x_0], label=f'Gamma={gamma}')
        axes1[1].plot(x_1, [w[0] + w[2]*x + w[4]*x**2 + w[6]*x**3 for x in x_1], label=f'Gamma={gamma}')
        
        # Calculate and plot MAP predictions
        map_w = map(phi, gamma, y_train)
        print("MAP Error:", ase(map_w, x_val, y_val))
        
        w = map_w.ravel()
        axes2[0].plot(x_0, [w[0] + w[1]*x + w[3]*x**2 + w[5]*x**3 for x in x_0], label=f'Gamma={gamma}')
        axes2[1].plot(x_1, [w[0] + w[2]*x + w[4]*x**2 + w[6]*x**3 for x in x_1], label=f'Gamma={gamma}')



    # Labeling Plots and saving
    plt.show()    
    plt.savefig('./problem_2_plots_MLE.pdf')
    
    plt.show()    
    plt.savefig('./problem_2_plots_MAP.pdf')



def hw2q2():
    Ntrain = 100
    data = generateData(Ntrain)
    plot3(data[0,:],data[1,:],data[2,:])
    xTrain = data[0:2,:]
    yTrain = data[2,:]
    
    Ntrain = 1000
    data = generateData(Ntrain)
    plot3(data[0,:],data[1,:],data[2,:])
    xValidate = data[0:2,:]
    yValidate = data[2,:]
    
    return xTrain,yTrain,xValidate,yValidate

def generateData(N):
    gmmParameters = {}
    gmmParameters['priors'] = [.3,.4,.3] # priors should be a row vector
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:,:,0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:,:,1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:,:,2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x,labels = generateDataFromGMM(N,gmmParameters)
    return x

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

def plot3(a,b,c,mark="o",col="b"):
  from matplotlib import pyplot
  import pylab
  from mpl_toolkits.mplot3d import Axes3D
  pylab.ion()
  fig = pylab.figure()
  ax = Axes3D(fig)
  ax.scatter(a, b, c,marker=mark,color=col)

  ax.set_xlabel("x1")
  ax.set_ylabel("x2")
  ax.set_zlabel("y")
  ax.set_title('Training Dataset')

def mle(phi, t):
    tphi = np.transpose(phi)
    pseudo_inv = np.linalg.inv(np.matmul(tphi, phi))
    results = np.matmul(pseudo_inv, tphi).dot(t)
    return results

def ase(w, x_val, y_val):
    N = len(y_val)
    x = [[1, x_val[0][i], x_val[1][i], x_val[0][i]**2, 
          x_val[1][i]**2, x_val[0][i]**3, x_val[1][i]**3] for i in range(N)]
    
    # Calculate the MSE
    total_error = sum((y_val[n] - np.dot(w, x[n]))**2 for n in range(N)) / N
    return round(total_error, 6)

def map(phi, gamma, t):
    tphi = np.transpose(phi)
    regularized_inv = np.linalg.inv(gamma * np.identity(phi.shape[1]) + np.matmul(tphi, phi))
    results = np.matmul(regularized_inv, tphi).dot(t)
    return results

def generate_features(x):
    return np.array([[1, x[0][i], x[1][i], x[0][i]**2, 
                      x[1][i]**2, x[0][i]**3, x[1][i]**3] for i in range(len(x[0]))])

if __name__ == '__main__':
    main()
