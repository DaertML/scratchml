import numpy as np

class Sigmoid():
    def forward(self, inputs):
        self.inputs = inputs
        self.inputs = np.clip(inputs, -500, 500)
        self.output = 1/(1+np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = 1/(1+np.exp(-dvalues)) * (1 - 1/(1+np.exp(-dvalues)))