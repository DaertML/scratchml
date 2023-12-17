import numpy as np

class Softplus():
    def __init__():
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.log(1+np.exp(inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues/(1+np.exp(-dvalues))
