import numpy as np

lambda_ = 1.0507
alpha = 1.67326

class Selu():
    def __init__():
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.output = lambda_ * np.where(inputs > 0, inputs, alpha * np.exp(inputs) - alpha)

    def backward(self, dvalues):
        self.dinputs = lambda_ * np.where(dvalues >= 0, 1, alpha * np.exp(dvalues))