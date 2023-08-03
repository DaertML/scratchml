import numpy as np

class Softmax:
    def __init__(self):
        pass

    def forward(self, inputs):
        #E = 2.71828182846
        #exp_vals = [E ** inp for inp in inputs]
        #norm_base = sum(exp_vals)
        #self.output = [exp/norm_base for exp in exp_vals]

        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values/np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)