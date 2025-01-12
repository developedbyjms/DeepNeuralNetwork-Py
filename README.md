# DeepNeuralNetwork-Py
DeepNeuralNetwork is a Python framework to build and train deep neural networks from scratch. It features customizable architectures, sigmoid activation, forward and backpropagation, and gradient descent. Designed for educational purposes, it helps users understand neural network mechanics without relying on external libraries.


## Overview
DeepNeuralNetwork is a Python-based framework for building and training deep neural networks from scratch. This project is designed to help users understand the core mechanics of neural networks, including forward propagation, backpropagation, and gradient descent, without relying on external machine learning libraries.

## Features
- **Customizable Architecture**: Define the number of input, hidden, and output layers to fit your dataset.
- **Activation Function**: Uses the sigmoid activation function and its derivative for backpropagation.
- **Training Algorithms**: Supports backpropagation with gradient descent.
- **Error Handling**: Provides error calculations for both output and hidden layers.
- **Educational Purpose**: Aimed at learners to help grasp neural network mechanics.

## Requirements
- Python 3.x
- NumPy for efficient matrix operations

## Installation
Clone the repository to your local machine:
```bash
git clone https://github.com/developedbyjms/DeepNeuralNetwork-Py.git

Navigate to the project directory:

cd DeepNeuralNetwork

Install the required libraries:

pip install numpy

Usage

Example Code

Here’s a quick example of how to use the DeepNeuralNetwork class:

import numpy as np
from DeepNeuralNetwork import DeepNeuralNetwork

# Create a dataset (XOR problem)
dataset = [
    (np.array([[0], [0]]), np.array([[0]])),
    (np.array([[0], [1]]), np.array([[1]])),
    (np.array([[1], [0]]), np.array([[1]])),
    (np.array([[1], [1]]), np.array([[0]]))
]

# Initialize the neural network
hidden_layers = [4]  # One hidden layer with 4 neurons
network = DeepNeuralNetwork(dataset, hidden_layers)

# Train the neural network
learning_rate = 0.1
epochs = 10000
network.train(learning_rate, epochs, randomly=False)

# Make a prediction
input_data = np.array([[1], [0]])  # Example input
output = network.feedforward(input_data)
print("Predicted Output:", output)

Methods

__init__(dataset, hidden_layers)

Initializes the neural network with a dataset and specified hidden layers.


feedforward(inputs)

Performs forward propagation through the network.


backPropagation(inputs, targets, learning_rate)

Calculates gradients, updates weights, and minimizes the error.


train(learning_rate, epoch, randomly)

Trains the network using backpropagation for the specified number of epochs.


Parameters

dataset: A list of tuples containing input and target output data.

hidden_layers: A list specifying the number of neurons in each hidden layer.

learning_rate: The rate at which weights are adjusted during training.

epoch: The number of times the training process iterates through the dataset.

randomly: Boolean indicating whether to train with randomized data.


Limitations

Only supports the sigmoid activation function.

Designed for small datasets; performance may degrade with large-scale datasets.

Requires manual data preprocessing (e.g., normalization).


Future Improvements

Add support for other activation functions (ReLU, Tanh).

Introduce advanced optimizers like Adam or RMSprop.

Implement support for mini-batch training.

Add visualization tools for loss and accuracy.


Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests for improvements or bug fixes.

License

This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgments

Inspired by the fundamental concepts of neural networks.

Built for educational and experimental use cases.


