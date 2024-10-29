import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def main():    
    d10k_validate = np.genfromtxt('d10k_validate.csv', delimiter=',').T
    d20_train = d10k_validate[:20]
    d200_train = d10k_validate[:200]
    d2000_train = d10k_validate[:2000]

    # Insert inputs for training
    X = np.array(d20_train)[:2]


    # Define parameter sizes
    nX = 2  # Number of input features
    nPerceptrons = 5  # Number of hidden layer perceptrons
    nY = 1  # Number of outputs
    size_params = (nX, nPerceptrons, nY)

    params_true = {
        'A': 0.3 * np.random.rand(nPerceptrons, nX),
        'b': 0.3 * np.random.rand(nPerceptrons, 1),
        'C': 0.3 * np.random.rand(nY, nPerceptrons),
        'd': 0.3 * np.random.rand(nY, 1)
    }

    Y = np.array(d10k_validate)[-1]

    # # Plotting the data in 3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(X[0, :], X[1, :], Y[0, :], color='g')
    # ax.set_xlabel('X_1')
    # ax.set_ylabel('X_2')
    # ax.set_zlabel('Y')
    # plt.title("Data visualization")
    # plt.show()

    # Initialize model parameters
    params = {
        'A': np.zeros_like(params_true['A']) + 0.1 * np.random.randn(nPerceptrons, nX),
        'b': np.zeros_like(params_true['b']) + 0.1 * np.random.randn(nPerceptrons, 1),
        'C': np.zeros_like(params_true['C']) + 0.1 * np.random.randn(nY, nPerceptrons),
        'd': np.mean(Y, axis=0, keepdims=True)  # initialize to mean of Y
    }

    # Flatten initialized parameters for optimization
    vec_params_init = np.hstack([params['A'].flatten(), params['b'].flatten(),
                                params['C'].flatten(), params['d'].flatten()])
    
    # Optimize model
    options = {'maxiter': 200 * len(vec_params_init), 'disp': True}
    result = minimize(lambda vecParams: objective_function(X, Y, size_params, vecParams),
                      vec_params_init, 
                      options=options)
    
    
    vecParams = result.x

    # Extract optimized parameters
    params["A"] = vecParams[:nX * nPerceptrons].reshape(nPerceptrons, nX)
    params["b"] = vecParams[nX * nPerceptrons:(nX + 1) * nPerceptrons].reshape(nPerceptrons, 1)
    params["C"] = vecParams[(nX + 1) * nPerceptrons:(nX + 1 + nY) * nPerceptrons].reshape(nY, nPerceptrons)
    params["d"] = vecParams[(nX + 1 + nY) * nPerceptrons:(nX + 1 + nY) * nPerceptrons + nY].reshape(nY, 1)
    
    # Visualize model output for training data
    H = mlp_model(X, params)
    plt.figure()
    plt.scatter(Y, H, color='r')
    plt.xlabel('Desired Output')
    plt.ylabel('Model Output')
    plt.title('Model Output Visualization For Training Data')
    plt.axis('equal')
    plt.show()

    # 3D plot to visualize Y and H together
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[0, :], X[1, :], Y[0, :], color='g', label='Y')
    ax.scatter(X[0, :], X[1, :], H[0, :], color='r', label='H')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('Y and H')
    ax.legend()
    plt.show()

    return vecParams, result.fun


def objective_function(X, Y, size_params, vec_params, use_cross_entropy=False):
    """
    Computes the objective function value for the MLP model.
    
    Parameters:
    X (ndarray): Input matrix of size (nX, N) where nX is the number of input features and N is the number of samples.
    Y (ndarray): Target matrix of size (nY, N) where nY is the number of outputs.
    size_params (tuple): Tuple containing (nX, nPerceptrons, nY) which specify the input, hidden, and output dimensions.
    vec_params (ndarray): Vector of parameters to be reshaped and assigned to the MLP model.
    use_cross_entropy (bool): If True, use cross-entropy objective; otherwise, use mean squared error (MSE). Defaults to False.

    Returns:
    float: The computed objective function value.
    """
    N = X.shape[1]  # number of samples
    nX, nPerceptrons, nY = size_params

    # Unpacking vec_params into weight matrices and bias vectors
    params = {
        'A': vec_params[0:nX * nPerceptrons].reshape(nPerceptrons, nX),
        'b': vec_params[nX * nPerceptrons + 1:(nX + 1) * nPerceptrons],
        'C': vec_params[(nX + 1) * nPerceptrons:(nX + 1 + nY) * nPerceptrons].reshape(nY, nPerceptrons),
        'd': vec_params[(nX + 1 + nY) * nPerceptrons:(nX + 1 + nY) * nPerceptrons + nY]
    }

        # 'A' : np.reshape(vecParams[0:nX * nPerceptrons], (nPerceptrons, nX))
        # 'b' : vecParams[nX * nPerceptrons:(nX + 1) * nPerceptrons]
        # 'C' : np.reshape(vecParams[(nX + 1) * nPerceptrons:(nX + 1 + nY) * nPerceptrons], (nY, nPerceptrons))
        # 'd' : vecParams[(nX + 1 + nY) * nPerceptrons:(nX + 1 + nY) * nPerceptrons + nY]

    # Get the model's output
    H = mlp_model(X, params)

    obj_fnc_value = -np.sum(Y * np.log(H + 1e-12)) / N

    return obj_fnc_value


def mlp_model(X, params):
    """
    Multi-layer perceptron (MLP) model function.

    Parameters:
    X (ndarray): Input matrix of size (nX, N), where nX is the number of input features and N is the number of samples.
    params (dict): Dictionary containing the parameters:
                   - 'A': Weight matrix for the hidden layer of size (nP, nX)
                   - 'b': Bias vector for the hidden layer of size (nP, 1)
                   - 'C': Weight matrix for the output layer of size (nY, nP)
                   - 'd': Bias vector for the output layer of size (nY, 1)
    use_softmax (bool): If True, apply softmax activation on the output layer. Defaults to False.

    Returns:
    ndarray: Output of the MLP model, either linear activations or softmax probabilities.
    """

    N = X.shape[1]  # number of samples
    nY = params['d'].shape[0]  # number of outputs
    
    # Compute hidden layer activations
    U = params['A'] @ X + params['b']  # u = Ax + b
    
    # Apply the activation function
    Z = activation_function(U)  # Activation function for hidden layer
    
    # Compute output layer activations
    V = params['C'] @ Z + params['d']  # v = Cz + d
    # N = X.shape[1]  # number of samples
    # nY = len(params['d'])  # number of outputs

    # # Hidden layer computations
    # U = params['A'] @ X + np.tile(params['b'], (1, N))  # u = Ax + b
    # Z = activation_function(U)  # Activation function for hidden layer

    # # Output layer computations
    # V = params['C'] @ Z + np.tile(params['d'], (1, N))  # v = Cz + d

    return V

def activation_function(input):
    # ISRU Sigmoid Style Nonlinearity
    out = np.divide(input, np.sqrt(1 + input ** 2))
    return out    


if __name__ == "__main__":
    main()