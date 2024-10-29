import numpy as np
import math
import utils
import matplotlib.pyplot as plt


def main(): 
    d10k_validate_x, d10k_validate_labels, gmmParameters = generateData(10000)
    d10k_validate = np.vstack((d10k_validate_x, d10k_validate_labels)).T
    np.savetxt('d10k_validate.csv', d10k_validate, delimiter=',')

    d20_train = generateData(20)
    np.savetxt('d20_train.csv', d20_train, delimiter=',')
    d200_train = generateData(200)
    np.savetxt('d200_train.csv', d200_train, delimiter=',')
    d2000_train = generateData(2000)
    np.savetxt('d2000_train.csv', d2000_train, delimiter=',')

    g1 = utils.evalGaussian(d10k_validate_x, 
                            gmmParameters['meanVectors'][:,0], 
                            gmmParameters['covMatrices'][:,:,0])
    
    g2 = utils.evalGaussian(d10k_validate_x, 
                            gmmParameters['meanVectors'][:,1], 
                            gmmParameters['covMatrices'][:,:,1])

    
     
    discriminant_scores_ERM = np.log(g2/g1)
    PfpERM, PtpERM, PerrorERM, threshold_listERM = ROC_Curve(discriminant_scores_ERM, d10k_validate_labels)

    gamma_theoretical = 0.6/0.4
    l_0 = np.where(d10k_validate_labels == 0)[1]
    l_1 = np.where(d10k_validate_labels == 1)[1]

    dis_0 = np.take(discriminant_scores_ERM, l_0)
    dis_1 = np.take(discriminant_scores_ERM, l_1)
    
    thy_lambda_0 = np.divide(len(np.where(dis_0 >= gamma_theoretical)[0]), len(dis_0)) # False Positives
    thy_lambda_1 = np.divide(len(np.where(dis_1 >= gamma_theoretical)[0]), len(dis_1)) # True Positives
    
    thy_p_err = thy_lambda_0 * 0.6 + (1 - thy_lambda_1) * 0.4

    print("Theoretical Minimum P(error): " + str(thy_p_err))
    print("Classifier Minimum P(error): " + str(min(PerrorERM)))

    plt.figure()
    plt.plot(PfpERM, PtpERM, '*m', label='ERM Classifier')
    plt.plot(thy_lambda_0, thy_lambda_1, '*b')
    plt.xlabel('False Positive Rate (Pfp)')
    plt.ylabel('True Positive Rate (Ptp)')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

    

def generateData(n): 
    gmmParameters = {}
    gmmParameters['priors'] = [0.6, 0.4]
    gmmParameters['meanVectors'] = np.array([[-0.9, 0.8, -1.1, 0.9], 
                                             [-1.1, 0.75, 0.9, -0.75]])
    
    gmmParameters['covMatrices'] = np.zeros((2, 2, 2))
    gmmParameters['covMatrices'][:,:,0] = np.array([[0.75, 0], [0, 1.25]])
    gmmParameters['covMatrices'][:,:,1] = np.array([[0.75, 0], [0, 1.25]])

    x, labels = generateDataFromGMM(n, gmmParameters)
    
    return x, labels, gmmParameters

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

def ROC_Curve(discriminant_scores, labels):
    sorted_scores = np.sort(discriminant_scores)
    eps = np.finfo(float).eps
    threshold_list = np.concatenate((
        [sorted_scores[0] - eps],
        (sorted_scores[:-1] + sorted_scores[1:]) / 2,
        [sorted_scores[-1] + eps]
    ))

    Ptn = np.zeros(len(threshold_list))
    Pfp = np.zeros(len(threshold_list)) 
    Ptp = np.zeros(len(threshold_list))
    Perror = np.zeros(len(threshold_list))

    for i in range(len(threshold_list)): 
        tau = threshold_list[i]
        decisions = np.greater_equal(discriminant_scores, tau)
        decisions = decisions.astype(float)

        Ptn[i] = np.divide(len(np.where((decisions == 0) & (labels == 0))[1]), len(np.where(labels == 0)[1]))
        Pfp[i] = np.divide(len(np.where((decisions == 1) & (labels == 0))[1]), len(np.where(labels == 0)[1]))
        Ptp[i] = np.divide(len(np.where((decisions == 1) & (labels == 1))[1]), len(np.where(labels == 1)[1]))
        Perror[i] = np.divide(len(np.where(decisions != labels)[1]), len(labels[0]))

    return Pfp, Ptp, Perror, threshold_list

if __name__ == "__main__":
    main()