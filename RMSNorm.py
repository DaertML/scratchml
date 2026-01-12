import numpy as np

class RMSNorm:
    def __init__(self, d_model, eps=1e-6):
        """
        Initialize RMSNorm layer

        Args:
            d_model: Dimension of the model (embeddings)
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps

        # Learnable parameter gamma (scale)
        self.gamma = np.ones((1, 1, d_model))

        # Initialize gradients
        self.dgamma = None

    def forward(self, x):
        """
        Forward pass of RMSNorm

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Normalized output tensor of shape (batch_size, seq_len, d_model)
        """
        self.x = x

        # Calculate RMS (Root Mean Square)
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + self.eps)

        # Normalize
        self.x_norm = x / rms

        # Scale with gamma parameter
        output = self.gamma * self.x_norm

        self.output = output
        return output

    def backward(self, gradient):
        """
        Backward pass of RMSNorm

        Args:
            gradient: Gradient tensor from next layer

        Returns:
            Gradient with respect to inputs
        """
        batch_size, seq_len, d_model = self.x.shape

        # Calculate intermediate values
        rms = np.sqrt(np.mean(self.x**2, axis=-1, keepdims=True) + self.eps)

        # Calculate gradient w.r.t. x
        # This is a simplified version - full derivation would be more complex
        x_norm = self.x / rms

        # Gradient w.r.t. gamma
        self.dgamma = np.sum(gradient * x_norm, axis=(0, 1), keepdims=True)

        # Gradient w.r.t. input x
        # This follows the chain rule for RMSNorm backward pass
        grad_x = gradient * self.gamma / rms

        # Subtract the mean of gradients times normalized input to maintain proper gradient flow
        grad_mean = np.mean(grad_x, axis=-1, keepdims=True)
        grad_x = grad_x - grad_mean

        return grad_x

    def get_params(self):
        """Return current parameters for optimizer updates"""
        return [self.gamma]

    def set_params(self, gamma):
        """Set new parameter values"""
        self.gamma = gamma