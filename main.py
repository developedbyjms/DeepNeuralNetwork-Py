from dnn import DeepNeuralNetwork


xor = [[[0, 0], [0]], [[0, 1], [1]],[[1, 0], [1]],[[1, 1], [0]]]

dnn = DeepNeuralNetwork(dataset = admission_data(), hidden_layers = [20, 10, 20])
dnn.train(learning_rate=0.10, epoch=10000, randomly=True)
print(dnn.feedforward([-1, -1]))



"""
    ~ DevelopedByJMS™
    ~ José Sixpenze
    ~ 12/01/2025
"""