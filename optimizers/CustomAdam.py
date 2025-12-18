class CustomAdam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment vector (mean of gradients)
        self.v = None  # Second moment vector (uncentered variance of gradients)
        self.t = 0     # Timestep

    def update_params(self, weights, gradients):
        # 1. Initialize moments on the first step
        if self.m is None:
            self.m = np.zeros_like(weights)
            self.v = np.zeros_like(weights)

        self.t += 1

        # 2. Update biased first and second moment estimates
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients**2)

        # 3. Compute bias-corrected first and second moment estimates
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # 4. Update weights
        weights += -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
