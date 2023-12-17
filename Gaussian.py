import numpy as np

class Gaussian():
    def __init__():
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.exp(-inputs**2)

    def backward(self, dvalues):
        self.dinputs = -2*dvalues*np.exp(-dvalues**2)