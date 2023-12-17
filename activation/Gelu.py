import numpy as np
#from scipy.stats import norm

def Gelu():
    def __init__():
        pass

    def forward(self, inputs):
        self.inputs = inputs
        self.output = inputs * 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (inputs + 0.044715 * inputs ** 3)))
        
    def backward(self, dvalues):
        # Using norm from scipy. Better to avoid dependencies apart from numpy
        #self.dinputs = 0.5 * norm.cdf(dvalues) + 0.5 * dvalues * norm.pdf(dvalues) + 0.5 * dvalues * norm.cdf(dvalues)

        self.dinputs = 0.5 * (1 + np.erf(dvalues / np.sqrt(2))) + dvalues * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * dvalues**2)
