# Neural Network from Scratch

This repository contains an implementation of a simple feedforward neural network in Python using NumPy. The network is designed to handle binary classification tasks.

## Table of Contents
- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Functions Explanation](#functions-explanation)
- [License](#license)

## Introduction
This project demonstrates the basic building blocks of a neural network, including forward propagation, backpropagation, and gradient descent. The code is intended for educational purposes and helps in understanding how neural networks work under the hood.

## Project Structure
- **`neural_network.py`**: The main script containing the neural network implementation.
- **`utils.py`**: A utility script containing helper functions like `sigmoid` and `relu`.
- **`Train_data.csv`**: A sample dataset used for training the network.

## Dataset
The dataset used for training consists of 13 features and binary labels (0 or 1). Each data point in the dataset represents a set of features that are used as input to the neural network, with the corresponding label indicating the class to which the data point belongs.

## Installation
To run this code, you'll need Python 3.x installed along with the following packages:
- `numpy`
- `pandas`
- `matplotlib`

You can install these dependencies using pip:
```bash
pip install numpy pandas matplotlib
```

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/neural-network-from-scratch.git
   ```
2. Navigate to the project directory:
   ```bash
   cd neural-network-from-scratch
   ```
3. Run the training script:
   ```bash
   python neural_network.py
   ```
   The script will train the neural network using the sample dataset and plot the cost function over epochs.

## Code Overview
The code is structured into several functions that handle different parts of the neural network's operations:

- **`initialize_parameters(layers_dims)`**: Initializes the weights and biases for each layer.
- **`linear_forward(A_prev, W, b)`**: Computes the linear part of a layer's forward propagation.
- **`linear_activation_forward(A_prev, W, b, activation_fn)`**: Applies the activation function after the linear transformation.
- **`L_model_forward(X, parameters)`**: Implements forward propagation for the entire network.
- **`compute_cost(AL, y)`**: Computes the cross-entropy cost.
- **`linear_backward(dZ, cache)`**: Implements the linear portion of backward propagation.
- **`activation_backward(dA, cache, activation_fn)`**: Backpropagates the error through the activation function.
- **`model_backward(AL, Y, caches)`**: Implements backward propagation for the entire network.
- **`update_parameters(parameters, grads, learning_rate)`**: Updates the model parameters using gradient descent.
- **`training(x, y, parameters, learning_rate, epochs)`**: Trains the neural network using the provided data.

## Functions Explanation
### `initialize_parameters(layers_dims)`
- Initializes the network's weights and biases based on the provided layer dimensions.

### `linear_forward(A_prev, W, b)`
- Computes the net input for a given layer during forward propagation.

### `linear_activation_forward(A_prev, W, b, activation_fn)`
- Applies the specified activation function (`relu` or `sigmoid`) to the linear output.

### `L_model_forward(X, parameters)`
- Implements the full forward pass through the network.

### `compute_cost(AL, y)`
- Calculates the cost using cross-entropy loss.

### `linear_backward(dZ, cache)`
- Computes the gradients of the loss with respect to the input, weights, and bias for a layer.

### `activation_backward(dA, cache, activation_fn)`
- Backpropagates through the activation function.

### `model_backward(AL, Y, caches)`
- Executes the backward pass for the entire network, computing gradients for all layers.

### `update_parameters(parameters, grads, learning_rate)`
- Updates the network's parameters using the computed gradients.

### `training(x, y, parameters, learning_rate, epochs)`
- Trains the neural network, updating parameters and tracking the cost over time.

## License
This project is licensed under the MIT License. Feel free to use and modify the code for your own purposes.