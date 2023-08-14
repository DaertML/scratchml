import numpy as np

class Tanh():
    def forward(self, inputs):
        self.inputs = inputs
        # This causes overflows, use log-sum-exp instead
        #self.output = (2/(1+np.exp(-2*inputs))) - 1

        # Compute the log of the sum of exponentials
        logsumexp = np.log(np.exp(inputs) + np.exp(-inputs))
        # Compute the tanh using the log-sum-exp trick
        self.output = (np.exp(inputs - logsumexp) - np.exp(-inputs - logsumexp)) / (np.exp(inputs - logsumexp) + np.exp(-inputs - logsumexp))
        return self.output
    
    def backward(self, dvalues):
        self.dinputs = 1 - self.forward(dvalues) ** 2