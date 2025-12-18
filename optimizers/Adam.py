import numpy as np
class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        # Initialize moment buffers as dictionaries/None to store per-layer state
        self.m_w = None
        self.v_w = None
        self.m_b = None
        self.v_b = None

    def _initialize_moments(self, layer):
        """Helper to initialize moments based on layer parameter shapes."""
        self.m_w = np.zeros_like(layer.weights)
        self.v_w = np.zeros_like(layer.weights)
        self.m_b = np.zeros_like(layer.bias)
        self.v_b = np.zeros_like(layer.bias)

    def _update_single_param(self, param, grad, m, v):
        """Applies the core Adam update logic to a single parameter group."""
        self.t += 1

        # 1. Update biased first and second moment estimates
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * (grad**2)

        # 2. Compute bias-corrected first and second moment estimates
        m_hat = m / (1 - self.beta1**self.t)
        v_hat = v / (1 - self.beta2**self.t)

        # 3. Update parameters
        param += -self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return m, v

    def update_params(self, layer):
        if self.m_w is None:
            self._initialize_moments(layer)
            # Reset timestep for the first call, it will be incremented inside _update_single_param
            self.t = 0 
        
        # NOTE: To correctly track the timestep 't' for bias correction, 
        # we update weights and then reset/re-increment 't' for biases, 
        # or structure the code to ensure 't' is incremented once per layer update.
        
        # Reset/Increment t once for the layer update
        self.t += 1
        current_t = self.t
        
        # --- Update Weights ---
        # 1. Update biased first and second moment estimates for weights
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * layer.dweights
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (layer.dweights**2)
        
        # 2. Compute bias-corrected estimates
        m_hat_w = self.m_w / (1 - self.beta1**current_t)
        v_hat_w = self.v_w / (1 - self.beta2**current_t)
        
        # 3. Apply update to weights
        layer.weights += -self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
        
        # --- Update Biases ---
        # 1. Update biased first and second moment estimates for biases
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * layer.dbiases
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (layer.dbiases**2)
        
        # 2. Compute bias-corrected estimates
        m_hat_b = self.m_b / (1 - self.beta1**current_t)
        v_hat_b = self.v_b / (1 - self.beta2**current_t)

        # 3. Apply update to biases
        layer.bias += -self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
