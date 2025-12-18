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

class CustomMuon:
    def __init__(self, learning_rate=0.02, momentum=0.95, ns_steps=5):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.ns_steps = ns_steps
        self.M = None  # Momentum buffer (First Moment)

    def update_params(self, weights, gradients):
        # 1. Initialize momentum on the first step
        if self.M is None:
            self.M = np.zeros_like(weights)

        # 2. Update biased first moment (momentum)
        # M_t = mu * M_{t-1} + (1 - mu) * G_t
        self.M = self.momentum * self.M + (1 - self.momentum) * gradients

        # 3. Compute the orthogonalized update (O_t)
        # Muon applies Newton-Schulz to the momentum matrix M_t
        O_t = newton_schulz_orthogonalize(self.M, steps=self.ns_steps)

        # 4. Update weights
        # W_t = W_{t-1} - eta_t * O_t
        weights += -self.learning_rate * O_t
