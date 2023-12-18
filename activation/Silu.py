import numpy as np

class Silu():
    def __init__(self):
        self.trainable = False

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs/(1+np.exp(-inputs))

    def backward(self, dvalues):
        neg_exp = np.exp(-dvalues)
        numerator = 1 + neg_exp * np.dot(dvalues, neg_exp)
        denominator = np.square(1+neg_exp)
        self.dinputs = numerator/denominator
