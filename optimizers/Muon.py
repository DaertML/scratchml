import numpy as np
# A simplified, conceptual Newton-Schulz iteration function
def newton_schulz_orthogonalize(G, steps=5, eps=1e-7):
    """
    Approximates Ortho(G) = U * V.T, where G = U * S * V.T (SVD).
    This process orthogonalizes the update matrix.
    NOTE: A full implementation is complex and requires matrix operations.
    """
    # Muon uses a polynomial approximation: X = aX + b(XX.T)X + c(XX.T)^2X
    # We use a conceptual placeholder for simplicity
    if G.ndim < 2 or G.size == 0:
        return G # Pass through for non-matrix tensors (e.g., biases)

    # 1. Normalize the matrix
    X = G / (np.linalg.norm(G, ord='fro') + eps) # Frobenius norm for normalization

    # 2. Apply Newton-Schulz iterations (simplified polynomial)
    a, b, c = 3.4445, -4.7750, 2.0315 # Typical optimized coefficients
    for _ in range(steps):
        # A = X @ X.T (Matrix multiplication)
        A = np.dot(X, X.T)

        # B = b * A + c * A @ A
        B = b * A + c * np.dot(A, A)

        # X = a * X + B @ X
        X = a * X + np.dot(B, X)

    return X # This X is the orthogonalized update matrix

class Muon:
    def __init__(self, learning_rate=0.02, momentum=0.95, ns_steps=5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.M_w = None  # Momentum buffer for weights
        self.M_b = None  # Momentum buffer for biases

    def _initialize_moments(self, layer):
        """Initializes moments based on layer parameter shapes."""
        self.M_w = np.zeros_like(layer.weights)
        self.M_b = np.zeros_like(layer.bias)

    def update_params(self, layer):
        if self.M_w is None:
            self._initialize_moments(layer)

        # --- Update Weights (Muon for 2D/4D Tensors) ---
        
        # 1. Store original shape for weights and flatten if necessary
        original_shape = layer.weights.shape
        is_matrix = layer.weights.ndim == 2
        
        # Muon requires a 2D matrix. Flatten (N, C, H, W) to (N, C*H*W).
        # We only flatten if it is a convolutional layer (ndim > 2).
        if not is_matrix:
            # Flatten the momentum buffer and gradients to 2D
            M_w_flat = self.M_w.reshape(original_shape[0], -1)
            dweights_flat = layer.dweights.reshape(original_shape[0], -1)
        else:
            # If it's already a matrix (e.g., fully connected layer), use as is
            M_w_flat = self.M_w
            dweights_flat = layer.dweights
        
        # 2. Update Momentum (M_w_flat)
        # M_t = mu * M_{t-1} + (1 - mu) * G_t
        M_w_flat = self.momentum * M_w_flat + (1 - self.momentum) * dweights_flat

        # 3. Orthogonalize the flattened momentum to get the update matrix (O_t)
        O_t_w_flat = newton_schulz_orthogonalize(M_w_flat, steps=self.ns_steps)

        # 4. Reshape the update and the momentum buffer back to original shape
        O_t_w = O_t_w_flat.reshape(original_shape)
        self.M_w = M_w_flat.reshape(original_shape)
        
        # 5. Apply update to weights (W_t = W_{t-1} - eta * O_t)
        layer.weights += -self.learning_rate * O_t_w

        # --- Update Biases (SGD with Momentum, as Biases are 1D) ---
        # 1. Update Momentum (M_b)
        self.M_b = self.momentum * self.M_b + (1 - self.momentum) * layer.dbiases
        
        # 2. Apply update to biases (B_t = B_{t-1} - eta * M_b)
        layer.bias += -self.learning_rate * self.M_b
