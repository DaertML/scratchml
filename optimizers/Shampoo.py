import numpy as np
from numpy.linalg import inv, svd

# --- Helper Function for Matrix Power (Shampoo requirement) ---
def _matrix_power(matrix, power):
    """
    Computes matrix^power using Singular Value Decomposition (SVD).
    
    Args:
        matrix (np.ndarray): The input matrix (assumed to be square and symmetric/PSD for this use case).
        power (float): The exponent to raise the matrix to.
        
    Returns:
        np.ndarray: The resulting matrix raised to the power.
    """
    # Use SVD for stability and for non-symmetric matrices if needed, though Shampoo's
    # preconditioners should be symmetric positive semi-definite (PSD).
    try:
        u, s, vh = svd(matrix)
        # Raise singular values to the power
        s_powered = np.diag(s**power)
        # Reconstruct the matrix: U * S^power * V^T
        return u @ s_powered @ vh
    except np.linalg.LinAlgError as e:
        # Handle cases where SVD fails (e.g., non-convergence)
        print(f"LinAlgError in _matrix_power: {e}")
        # Return identity matrix for safe fallback, though this is a simplification
        return np.eye(matrix.shape[0])


# --- Shampoo Optimizer Class ---
class Shampoo:
    def __init__(self, learning_rate=1e-3, epsilon=1e-6, update_freq=10):
        self.learning_rate = learning_rate
        # Epsilon for numerical stability
        self.epsilon = epsilon
        # Frequency to compute the expensive fractional matrix inverse power
        self.update_freq = update_freq
        self.step_count = 0
        # State dictionary to store preconditioners (L and R) and their inverses (L_inv_root, R_inv_root)
        self.state = {}

    def update_params(self, layer):
        # Shampoo is typically applied to the weight matrix (W) of a layer.
        # It's generally not used for biases, which are often updated with standard AdaGrad/Adam
        # or simply SGD, but we include a simplified bias update for completeness/architecture matching.

        # 1. Increment step count
        self.step_count += 1
        
        # 2. Initialize state for the layer if not present
        if layer not in self.state:
            # Assumes weights are 2D (matrix) of shape (m, n) and dweights is the gradient G
            m, n = layer.weights.shape
            
            # Left preconditioner L (m x m) - initialized with epsilon * I
            self.state[layer] = {
                'L': self.epsilon * np.eye(m), 
                'R': self.epsilon * np.eye(n),
                'L_inv_root': np.eye(m),
                'R_inv_root': np.eye(n)
            }
        
        state = self.state[layer]
        G = layer.dweights # Gradient of the weights

        # 3. Update Preconditioners (L and R)
        # L_t = L_{t-1} + G_t G_t^T
        state['L'] += G @ G.T
        # R_t = R_{t-1} + G_t^T G_t
        state['R'] += G.T @ G

        # 4. Compute Fractional Matrix Inverse Power (L_inv_root and R_inv_root) periodically
        # The computation of L^{-1/4} and R^{-1/4} is the most expensive part.
        if self.step_count % self.update_freq == 0:
            # Compute L^{-1/4}
            state['L_inv_root'] = _matrix_power(state['L'], -0.25) # -1/4
            # Compute R^{-1/4}
            state['R_inv_root'] = _matrix_power(state['R'], -0.25) # -1/4

        # 5. Compute the preconditioned gradient (Search Direction)
        # Search_Direction = L^{-1/4} * G * R^{-1/4}
        # Note: L^{-1/4} * G is matrix multiplication. G * R^{-1/4} is an element-wise product if R^{-1/4}
        # was a vector/diagonal, but here it's matrix multiplication.
        # NumPy's @ operator handles this for 2D arrays (matrices).
        
        # Intermediate: L_inv_root @ G (m x n)
        # Final: (L_inv_root @ G) @ R_inv_root (m x n)
        preconditioned_gradient = state['L_inv_root'] @ G @ state['R_inv_root']
        
        # 6. Update parameters
        # W_{t+1} = W_t - eta * Search_Direction
        layer.weights += -self.learning_rate * preconditioned_gradient

        # Biases: Apply a simple AdaGrad-like update or SGD (simplified here for architecture consistency)
        # For simplicity and to match the base architecture, we'll apply an SGD-like update to bias.
        # In a full implementation, you'd typically use a separate, standard adaptive method for bias.
        if layer.bias is not None:
             layer.bias += -self.learning_rate * layer.dbiases
