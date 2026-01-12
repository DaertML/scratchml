import numpy as np

class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        """
        Initialize LayerNorm layer

        Args:
            d_model: Dimension of the model (embeddings)
            eps: Small constant for numerical stability
        """
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters gamma (scale) and beta (shift)
        self.gamma = np.ones((1, 1, d_model))
        self.beta = np.zeros((1, 1, d_model))

        # Initialize gradients
        self.dgamma = None
        self.dbeta = None

    def forward(self, x):
        """
        Forward pass of LayerNorm

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Normalized output tensor of shape (batch_size, seq_len, d_model)
        """
        self.x = x

        # Calculate mean and variance
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)

        # Normalize
        x_norm = (x - mean) / np.sqrt(var + self.eps)

        # Scale with gamma and shift with beta
        output = self.gamma * x_norm + self.beta

        self.output = output
        return output

    def backward(self, gradient):
        """
        Backward pass of LayerNorm

        Args:
            gradient: Gradient tensor from next layer

        Returns:
            Gradient with respect to inputs
        """
        batch_size, seq_len, d_model = self.x.shape

        # Calculate mean and variance again for gradient computation
        mean = np.mean(self.x, axis=-1, keepdims=True)
        var = np.var(self.x, axis=-1, keepdims=True)

        # Standard deviation with epsilon
        std_inv = 1.0 / np.sqrt(var + self.eps)

        # Normalize the input for gradient calculation
        x_norm = (self.x - mean) * std_inv

        # Gradient w.r.t. gamma and beta
        self.dgamma = np.sum(gradient * x_norm, axis=(0, 1), keepdims=True)
        self.dbeta = np.sum(gradient, axis=(0, 1), keepdims=True)

        # Gradient w.r.t. input
        # This follows the chain rule for LayerNorm backward pass
        d_x_norm = gradient * self.gamma
        d_var = np.sum(d_x_norm * (self.x - mean) * -0.5 * std_inv**3, axis=-1, keepdims=True)
        d_mean = np.sum(d_x_norm * -std_inv, axis=-1, keepdims=True) + d_var * np.mean(-2 * (self.x - mean), axis=-1, keepdims=True)

        # Final gradient
        grad_x = d_x_norm * std_inv + d_var * 2 * (self.x - mean) / (batch_size * seq_len) + d_mean / (batch_size * seq_len)

        return grad_x

    def get_params(self):
        """Return current parameters for optimizer updates"""
        return [self.gamma, self.beta]

    def set_params(self, gamma, beta):
        """Set new parameter values"""
        self.gamma = gamma
        self.beta = beta