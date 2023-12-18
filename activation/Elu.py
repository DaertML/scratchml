import numpy as np

class Elu():
    def __init__(self, alpha):
        self.alpha = alpha
        self.trainable = False
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * np.exp(inputs) - self.alpha)

    def backward(self, dvalues):
        self.dinputs = np.where(dvalues > 0, 1, self.alpha * np.exp(dvalues))
