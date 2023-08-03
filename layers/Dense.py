import numpy as np
np.random.seed(0)

class Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.bias = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.bias

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

if __name__ == "__main__":
    d1 = Dense(3,5)
    inputs = [[1,2,3],
          [4,5,6],
          [7,8,9]]
    print(d1.weights, d1.bias)
    d1.forward(inputs)
    print(d1.output)