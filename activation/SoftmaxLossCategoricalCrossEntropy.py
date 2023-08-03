import numpy as np
import sys
sys.path.append("..")
from activation import Softmax
from loss import CategoricalCrossEntropy

class SoftmaxLossCategoricalCrossEntropy():
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        
        self.dinputs = np.array(dvalues.copy())
        self.dinputs[list(range(samples)), y_true] -= 1
        self.dinputs = self.dinputs/samples