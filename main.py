from dnn import DeepNeuralNetwork


xor = [[[0, 0], [0]], [[0, 1], [1]],[[1, 0], [1]],[[1, 1], [0]]]

dnn = DeepNeuralNetwork(dataset = xor, hidden_layers = [4, 2, 4])
dnn.train(learning_rate=0.10, epoch=1000, randomly=True)
print(dnn.feedforward([-1, -1]))



"""
    ~ DevelopedByJMS™
    ~ José Sixpenze
    ~ 12/01/2025
"""
