# -*-coding:utf8;-*-
import numpy as np
import math, sys, os, time

""" 
This script defines a series of functions to perform operations commonly used 
in neural networks, including the sigmoid activation function, its derivative, 
error calculation, gradient computation, and weight adjustment. 
"""

# Define the sigmoid activation function and its derivative using lambda functions.
sig = lambda x: 1 / (1 + math.exp(-x))  # Sigmoid function
sig_der = lambda x: x * (1 - x)  # Derivative of the sigmoid function


""" CALCULATE SIGMOID """
def sigmoid(x):
    """
    Applies the sigmoid activation function to a scalar, list, or matrix.

    Parameters:
    - x: A scalar, 1D list, or 2D list (matrix).

    Returns:
    - The sigmoid of x.
    - Handles nested lists (e.g., 2D arrays) and returns the appropriate structure.
    """
    try:
        return sig(x)  # Scalar case
    except:
        try:
            return list(map(sig, x))  # 1D list case
        except:
            try:
                result = []
                for rows in x:  # 2D list case
                    result.append(list(map(sig, rows)))
                return result
            except:
                return "error_007"  # Error handling for unsupported input types


""" CALCULATE THE DERIVATIVE OF SIGMOID """
def sigmoid_derivate(x):
    """
    Applies the derivative of the sigmoid function to a scalar, list, or matrix.

    Parameters:
    - x: A scalar, 1D list, or 2D list (matrix).

    Returns:
    - The derivative of the sigmoid function for x.
    - Handles nested lists (e.g., 2D arrays) and returns the appropriate structure.
    """
    try:
        return sig_der(x)  # Scalar case
    except:
        try:
            return list(map(sig_der, x))  # 1D list case
        except:
            try:
                result = []
                for rows in x:  # 2D list case
                    result.append(list(map(sig_der, rows)))
                return result
            except:
                return []  # Return an empty list if input type is unsupported

""" CALCULATE SQUARED ERROR """
def squared_error(data):
    """
    Computes the absolute value of errors (squared error) for a list or matrix.

    Parameters:
    - data: A list or 2D list (matrix) of errors.

    Returns:
    - A list of absolute error values.
    """
    try:
        return list(map(lambda x: abs(x), data))  # 1D list case
    except:
        return [list(map(lambda x: abs(x), array)) for array in data]  # 2D list case

""" FORWARD PROPAGATION: CALCULATE ACTIVATIONS """
def calculate_result(weight, data, bias):
    """
    Computes the result of forward propagation for a layer.

    Parameters:
    - weight: A matrix of weights for the layer.
    - data: Input data as a vector.
    - bias: A vector of biases for the layer.

    Returns:
    - Activated output of the layer.
    """
    return sigmoid(np.add(np.dot(weight, data), bias).tolist())

""" CALCULATE OUTPUT LAYER ERROR """
def calculate_output_error(targets, ai_result):
    """
    Computes the error for the output layer.

    Parameters:
    - targets: The expected outputs (target values).
    - ai_result: The actual outputs from the neural network.

    Returns:
    - A list of error values.
    """
    return np.subtract(targets, ai_result).tolist()

""" CALCULATE HIDDEN LAYER ERROR """
def calculate_hidden_error(weight, forward_error):
    """
    Computes the error for a hidden layer.

    Parameters:
    - weight: Weights of the connections leading to this layer.
    - forward_error: Error from the next layer (output or another hidden layer).

    Returns:
    - A list of error values for the hidden layer.
    """
    return np.dot(np.array(weight).T, forward_error).tolist()

""" CALCULATE OUTPUT GRADIENT """
def calculate_output_graddient(output_error, output_result, learning_rate):
    """
    Computes the gradient for the output layer.

    Parameters:
    - output_error: The error for the output layer.
    - output_result: The output of the layer.
    - learning_rate: The learning rate for training.

    Returns:
    - The gradient for the output layer.
    """
    return np.multiply(np.multiply(output_error, sigmoid_derivate(output_result)), learning_rate).tolist()

""" CALCULATE HIDDEN LAYER GRADIENT """
def calculate_hidden_graddient(hidden_error, hidden_result, learning_rate):
    """
    Computes the gradient for a hidden layer.

    Parameters:
    - hidden_error: The error for the hidden layer.
    - hidden_result: The output of the hidden layer.
    - learning_rate: The learning rate for training.

    Returns:
    - The gradient for the hidden layer.
    """
    return np.multiply(np.multiply(hidden_error, sigmoid_derivate(hidden_result)), learning_rate).tolist()

""" CALCULATE NEW WEIGHT """
def calculate_new_weight(graddient, _input):
    """
    Computes the weight updates for a layer.

    Parameters:
    - graddient: The gradient for the layer.
    - _input: The input to the layer.

    Returns:
    - The new weights for the layer.
    """
    return np.multiply(graddient, np.array(_input).T).tolist()

""" ADJUST WEIGHT """
def adjust_weight(old_weight, new_weight):
    """
    Updates the weights by adding the new weight adjustments.

    Parameters:
    - old_weight: The current weights of the layer.
    - new_weight: The weight adjustments (deltas).

    Returns:
    - Updated weights.
    """
    return np.add(old_weight, new_weight).tolist()
    
    
"""
    ~ DevelopedByJMS™
    ~ José Sixpenze
    ~ 12/01/2025
"""
