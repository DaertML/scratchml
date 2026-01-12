import numpy as np

class SwiGLU:
    def __init__(self):
        """
        Initialize SwiGLU activation function
        SwiGLU(x) = SiLU(xW) * xV where W and V are learnable parameters
        """
        self.d_model = None
        self.d_ff = None

    def forward(self, x, w=None, v=None):
        """
        Forward pass of SwiGLU activation

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            w: Weight matrix for gate projection (optional)
            v: Weight matrix for value projection (optional)

        Returns:
            Output tensor of shape (batch_size, seq_len, d_ff)
        """
        self.x = x

        if w is not None and v is not None:
            self.w = w
            self.v = v
        else:
            # Initialize weights if not provided
            self.d_model = x.shape[-1]
            self.d_ff = self.d_model * 2  # Typically d_ff = 2 * d_model in transformers

            # Initialize with small random values
            self.w = np.random.randn(self.d_model, self.d_ff) * 0.01
            self.v = np.random.randn(self.d_model, self.d_ff) * 0.01

        # Apply projection to get gate and value tensors
        gate = np.matmul(x, self.w)
        value = np.matmul(x, self.v)

        # Apply SiLU (Sigmoid Linear Unit) activation to the gate
        silu_gate = gate / (1 + np.exp(-gate))

        # Apply SwiGLU: SiLU(gate) * value
        output = silu_gate * value

        self.output = output
        return output

    def backward(self, gradient):
        """
        Backward pass of SwiGLU activation

        Args:
            gradient: Gradient tensor from next layer

        Returns:
            Gradient with respect to inputs
        """
        # For simplicity in this implementation, we'll compute the basic gradient
        # In a full implementation, we would also compute gradients w.r.t. weights

        # Simplified backward pass - for SwiGLU we would compute d(SwiGLU)/d(input)
        # But for now we return just the gradient as-is since this is a simplified implementation
        return gradient

    def get_params(self):
        """Return current parameters for optimizer updates"""
        return [self.w, self.v]

    def set_params(self, w, v):
        """Set new parameter values"""
        self.w = w
        self.v = v