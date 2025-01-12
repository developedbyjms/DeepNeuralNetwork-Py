import deepmath as dm
import random as rm

# Define the DeepNeuralNetwork class for creating and training a neural network.
class DeepNeuralNetwork:
    def __init__(self, dataset=[], hidden_layers=[]):
        """
        Initializes the neural network structure, weights, and biases.

        Parameters:
        - dataset: A list of tuples, where each tuple consists of input data (X) and target output (Y).
        - hidden_layers: A list of integers defining the number of neurons in each hidden layer.
        
        Steps:
        - Defines the structure of the network based on input, hidden, and output layers.
        - Randomly initializes weights and biases for all layers.
        """
        # Initialize the network structure (layers).
        self.layers = [len(dataset[0][0])]  # Input layer size
        for n in hidden_layers:
            self.layers.append(n)  # Add hidden layers
        self.layers.append(len(dataset[0][1]))  # Output layer size
        print(self.layers)  # Print the layer structure

        self.dataset = dataset  # Store the dataset
        self.weights = []  # List to store weights for each layer
        self.biases = []  # List to store biases for each layer

        # Initialize weights and biases for each layer
        for i in range(len(self.layers) - 1):
            self.weights.append(dm.random_matrix(self.layers[i + 1], self.layers[i]))
            self.biases.append(dm.random_matrix(self.layers[i + 1], 1))

    def feedforward(self, inputs):
        """
        Performs the feedforward operation to compute outputs for given inputs.

        Parameters:
        - inputs: Input data as a list or array.

        Returns:
        - Output of the final layer after feedforward computation.
        """
        # Compute the output of the first layer
        results = [dm.feedforward(self.weights[0], dm.from_array(inputs), self.biases[0])]

        # Compute the output of subsequent layers
        for i in range(len(self.layers) - 2):
            results.append(dm.feedforward(self.weights[i + 1], results[i], self.biases[i + 1]))

        return results[-1]  # Return the output of the last layer

    def backPropagation(self, inputs, targets, learning_rate):
        """
        Performs the backpropagation algorithm to adjust weights and biases.

        Parameters:
        - inputs: Input data as a list or array.
        - targets: Target outputs corresponding to the inputs.
        - learning_rate: Learning rate for weight updates.

        Returns:
        - A list containing results, errors, gradients, delta weights, and updated weights.
        """
        # Forward pass to compute activations
        results = [dm.feedforward(self.weights[0], dm.from_array(inputs), self.biases[0])]
        for i in range(len(self.layers) - 2):
            results.append(dm.feedforward(self.weights[i + 1], results[i], self.biases[i + 1]))

        # Compute output error
        errors = [dm.output_error(dm.from_array(targets), results[-1])]

        # Compute errors for hidden layers (backward pass)
        for i in range(len(results) - 1):
            errors.append(dm.hidden_error(self.weights[-1 - i], errors[i]))

        # Compute gradients for each layer
        gradients = []
        for i in range(len(results)):
            gradients.append(dm.graddient(errors[i], results[-1 - i], learning_rate))

        # Compute delta weights for each layer
        delta_weights = []
        for i in range(len(self.weights) - 1):
            delta_weights.append(dm.delta_weights(gradients[i], results[-2 - i]))
        delta_weights.append(dm.delta_weights(gradients[-1], dm.from_array(inputs)))

        # Update weights using delta weights
        for i in range(len(self.weights)):
            self.weights[-1 - i] = dm.adjust_weights(self.weights[-1 - i], delta_weights[i])

        return [results, errors, gradients, delta_weights, self.weights]

    def train(self, learning_rate, epoch, randomly):
        """
        Trains the neural network using the dataset.

        Parameters:
        - learning_rate: Learning rate for weight updates.
        - epoch: Number of epochs to train the model.
        - randomly: Boolean to indicate if training samples should be chosen randomly.

        Steps:
        - If randomly is True, select random samples for each epoch.
        - If randomly is False, train on all samples sequentially.
        - Prints the epoch, predictions, expected outputs, and errors for each step.
        """
        if randomly:
            for ep in range(epoch):
                # Select a random data point from the dataset
                chose_data = rm.choice(self.dataset)
                process = self.backPropagation(chose_data[0], chose_data[1], learning_rate)

                # Print progress
                print("=" * 50)
                print("EPOCH: ", ep + 1, "/", epoch)
                print("=" * 50)
                print("PREDICTED: ", process[0][-1], "\nEXPECTED : ", chose_data[1])
                print("ERROR:", process[1][0])
                print("=" * 50, "\n")
        else:
            for ep in range(epoch):
                for i in range(len(self.dataset)):
                    process = self.backPropagation(self.dataset[i][0], self.dataset[i][1], learning_rate)

                    # Print progress
                    print("=" * 50)
                    print("EPOCH: ", ep + 1, "/", epoch, "  ~  DATA: ", i + 1)
                    print("=" * 50)
                    print("PREDICTED: ", process[0][-1], "\nEXPECTED : ", self.dataset[i][1])
                    print("ERROR:", process[1][0])
                    print("=" * 50, "\n")
                    
                    
"""
    ~ DevelopedByJMS™
    ~ José Sixpenze
    ~ 12/01/2025
"""





