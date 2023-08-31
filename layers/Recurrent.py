import numpy as np

class Recurrent:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize the weights randomly
        self.W_xh = np.random.randn(input_size, hidden_size) * 0.01
        self.W_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_hy = np.random.randn(hidden_size, output_size) * 0.01
        
        # Initialize the hidden state to zeros
        self.h = np.zeros((1, hidden_size))
    
    def forward(self, x):
        # x: input vector of shape (1, input_size)
        
        # Compute the new hidden state
        self.h = np.dot(x, self.W_xh) + np.dot(self.h, self.W_hh)
        y = np.dot(self.h, self.W_hy)
        
        self.output = y
        return y
    
    def backward(self, x, y, y_true):
        # Compute the gradients of the loss with respect to y
        dy = y - y_true
        
        # Compute the gradients of the loss with respect to W_hy
        dW_hy = np.dot(self.h.T, dy)
        
        # Compute the gradients of the loss with respect to h
        dh = np.dot(dy, self.W_hy.T)

        # Compute the gradients of the loss with respect to W_hh
        dW_hh = np.dot(self.h.T, dh)
        
        # Compute the gradients of the loss with respect to W_xh
        dW_xh = np.dot(x.T, dh)
        
        return dW_xh, dW_hh, dW_hy
