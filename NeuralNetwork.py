import pandas as pd
import numpy as np
import utils
import matplotlib.pyplot as plt

def initialize_parameters(layers_dims):
    """
    Initializes the weights and biases for each layer of the neural network.

    Parameters:
    layers_dims -- list containing the number of neurons in each layer

    Returns:
    parameters -- python dictionary containing the initialized weights and biases
    """
    np.random.seed(1)  # Set a seed for reproducibility
    parameters = {}
    L = len(layers_dims)  # Number of layers in the network

    for l in range(1, L):
        # Initialize weights with small random values
        parameters["W" + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 0.01
        # Initialize biases with zeros
        parameters["b" + str(l)] = np.zeros((layers_dims[l], 1))
    
    return parameters

def linear_forward(A_prev, W, b):
    """
    Implements the linear part of a layer's forward propagation.

    Parameters:
    A_prev -- activations from the previous layer
    W -- weights matrix of the current layer
    b -- bias vector of the current layer

    Returns:
    Z -- the input of the activation function (pre-activation parameter)
    cache -- a tuple containing "A_prev", "W", and "b" for backward pass
    """
    Z = np.dot(W, A_prev) + b  # Compute the linear transformation
    cache = (A_prev, W, b)  # Store values for backward propagation
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation_fn):
    """
    Implements the forward propagation for the LINEAR->ACTIVATION layer.

    Parameters:
    A_prev -- activations from the previous layer
    W -- weights matrix of the current layer
    b -- bias vector of the current layer
    activation_fn -- the activation function to be used ("sigmoid" or "relu")

    Returns:
    A -- the output of the activation function
    cache -- a tuple containing "linear_cache" and "activation_cache" for backward pass
    """
    if activation_fn == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)  # Linear forward step
        A, activation_cache = utils.sigmoid(Z)  # Apply sigmoid activation
    elif activation_fn == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)  # Linear forward step
        A, activation_cache = utils.relu(Z)  # Apply relu activation
    
    cache = (linear_cache, activation_cache)  # Store caches for backward pass
    
    return A, cache

def L_model_forward(X, parameters, hidden_layers_activation_fn="relu"):
    """
    Implements forward propagation for the entire network.

    Parameters:
    X -- input data
    parameters -- python dictionary containing the network's parameters
    hidden_layers_activation_fn -- activation function to be used for hidden layers ("relu" by default)

    Returns:
    AL -- the output of the final layer
    caches -- list of caches containing the values for backward pass
    """
    A = X
    caches = []
    L = len(parameters) // 2  # Number of layers in the network

    # Implement forward propagation for hidden layers
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters["W" + str(l)], parameters["b" + str(l)], activation_fn=hidden_layers_activation_fn)
        caches.append(cache)

    # Implement forward propagation for the output layer using sigmoid activation
    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation_fn="sigmoid")
    caches.append(cache)
    
    return AL, caches

def compute_cost(AL, y):
    """
    Computes the cost function.

    Parameters:
    AL -- probability vector corresponding to the predictions
    y -- true labels vector

    Returns:
    cost -- cross-entropy cost
    """
    m = y.shape[1]  # Number of examples
    # Compute the cross-entropy cost
    cost = (-1/m)*(np.dot(y,np.log(AL.T)) + np.dot((1-y),np.log(1-AL.T)))
    
    return cost

def linear_backword(dZ, cache):
    """
    Implements the linear portion of backward propagation for a single layer.

    Parameters:
    dZ -- gradient of the cost with respect to the linear output
    cache -- tuple containing A_prev, W, b from forward propagation

    Returns:
    dA_prev -- gradient of the cost with respect to the activation of the previous layer
    dW -- gradient of the cost with respect to W
    db -- gradient of the cost with respect to b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]  # Number of examples

    # Compute gradients using the chain rule
    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

def activation_backward(dA, cache, activation_fn):
    """
    Implements the backward propagation for the ACTIVATION->LINEAR layer.

    Parameters:
    dA -- post-activation gradient for current layer
    cache -- tuple containing linear_cache and activation_cache
    activation_fn -- the activation function to be used ("sigmoid" or "relu")

    Returns:
    dA_prev -- gradient of the cost with respect to the activation of the previous layer
    dW -- gradient of the cost with respect to W
    db -- gradient of the cost with respect to b
    """
    linear_cache, activation_cache = cache

    if activation_fn == "sigmoid":
        dZ = utils.sigmoid_backward(dA, activation_cache)  # Compute the gradient for sigmoid
    elif activation_fn == "relu":
        dZ = utils.relu_backward(dA, activation_cache)  # Compute the gradient for relu
    
    dA_prev, dW, db = linear_backword(dZ, linear_cache)  # Compute the gradients for the current layer
    
    return dA_prev, dW, db

def model_backward(AL, Y, caches):
    """
    Implements the backward propagation for the entire network.

    Parameters:
    AL -- probability vector, output of the forward propagation (L_model_forward)
    Y -- true labels vector
    caches -- list of caches containing the values from forward propagation

    Returns:
    grads -- dictionary with the gradients for each layer
    """
    grads = {}
    L = len(caches)  # Number of layers
    Y = Y.reshape(AL.shape)  # Ensure Y is the same shape as AL

    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))  # Derivative of the cost function with respect to AL

    # Compute gradients for the output layer
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = activation_backward(dAL, current_cache, "sigmoid")

    # Compute gradients for the hidden layers (in reverse order)
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activation_backward(grads["dA" + str(l + 1)], current_cache, "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    
    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Updates the parameters using gradient descent.

    Parameters:
    parameters -- python dictionary containing the network's parameters
    grads -- python dictionary containing the gradients
    learning_rate -- learning rate used in the update rule

    Returns:
    parameters -- updated parameters
    """
    L = len(parameters) // 2  # Number of layers in the neural network

    # Update each parameter (W and b)
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    
    return parameters

def training(x, y, parameters, learning_rate, epochs):
    """
    Trains the neural network.

    Parameters:
    x -- input data
    y -- true labels
    parameters -- initialized parameters of the network
    learning_rate -- learning rate for gradient descent
    epochs -- number of iterations

    Returns:
    cost -- list containing the cost at each epoch
    parameters -- updated parameters after training
    """
    cost = []  # Initialize a list to store the cost at each epoch

    # Training loop
    for i in range(epochs):
        AL, cache = L_model_forward(x, parameters, "sigmoid")  # Forward pass
        cost.append(compute_cost(AL, y))  # Compute the cost
        grades = model_backward(AL, y, cache)  # Backward pass (compute gradients)
        parameters = update_parameters(parameters, grades, learning_rate)  # Update the parameters
    
    return cost, parameters

def train():
    """
    Loads the dataset, initializes parameters, and trains the model.
    """
    # Load the training dataset
    train_dataset = pd.read_csv(r"Train_data.csv")
    # Extract features and labels from the dataset
    x = np.array(train_dataset.iloc[:, 1:14]).reshape(13, 97)
    y = np.array(train_dataset.iloc[:, 0]).reshape(1, 97)
    print("x.shape", x.shape, "y.shape", y.shape)
    
    # Initialize parameters
    parameters = initialize_parameters([13, 2, 2, 1])
    epochs = 1000
    learning_rate = 0.01
    
    # Train the model
    cost, parameters = training(x, y, parameters, learning_rate, epochs)
    
    # Plot the cost function
    cost = np.array(cost).reshape(epochs, 1)
    print(cost)
    plt.plot(cost)
    plt.show()

train()  # Execute the training function
