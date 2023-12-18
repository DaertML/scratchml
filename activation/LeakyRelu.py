import numpy as np

class LeakyRelu():
    def __init__(self):
        self.trainable = False

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, 0.01*inputs)

    def backward(self, dvalues):
        self.dinputs = np.where(dvalues > 0, 1, 0.01)
