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
class ShampooCNN:
    def __init__(self, learning_rate=1e-3, epsilon=1e-6, update_freq=10):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.update_freq = update_freq
        self.step_count = 0
        self.state = {}

    def update_params(self, layer):
        self.step_count += 1
        
        # Determine the shape and matricization strategy
        original_shape = layer.weights.shape
        num_dims = len(original_shape)
        
        if num_dims == 2:
            # Case 1: Standard Dense/Linear Layer (2D) - Keep as is
            G = layer.dweights
            m, n = original_shape
            
        elif num_dims == 4:
            # Case 2: Convolutional Layer (4D) - Matricize the tensor
            # Conv weights shape: (k_h, k_w, C_in, C_out)
            # Matricization for Shampoo (Kronecker-factored)
            # Flatten (k_h * k_w * C_in) into the input dimension (n)
            # Keep C_out as the output dimension (m)
            
            # New dimensions: m = C_out, n = k_h * k_w * C_in
            m = original_shape[3] # Output channels
            n = original_shape[0] * original_shape[1] * original_shape[2] # Kernel size * Input channels

            # Reshape (Matricize): move output channels to the first dimension, then flatten the rest
            G = layer.dweights.transpose(3, 0, 1, 2).reshape(m, n)
            W_flat = layer.weights.transpose(3, 0, 1, 2).reshape(m, n)
            
        else:
            # Fallback for other tensor types (e.g., embeddings, but usually not supported)
            # For simplicity, we skip non-2D/4D layers or apply SGD.
            print(f"Shampoo does not support {num_dims}D tensor yet. Applying SGD to {layer}")
            layer.weights += -self.learning_rate * layer.dweights
            if layer.bias is not None:
                layer.bias += -self.learning_rate * layer.dbiases
            return

        # --- Shampoo Core Logic (Applies to both 2D and 4D matricized tensors) ---

        if layer not in self.state:
            # Initialize preconditioners L (m x m) and R (n x n)
            self.state[layer] = {
                'L': self.epsilon * np.eye(m), 
                'R': self.epsilon * np.eye(n),
                'L_inv_root': np.eye(m),
                'R_inv_root': np.eye(n)
            }
        
        state = self.state[layer]

        # 1. Update Preconditioners
        state['L'] += G @ G.T
        state['R'] += G.T @ G

        # 2. Compute Fractional Matrix Inverse Power (Periodically)
        if self.step_count % self.update_freq == 0:
            state['L_inv_root'] = _matrix_power(state['L'], -0.25)
            state['R_inv_root'] = _matrix_power(state['R'], -0.25)

        # 3. Compute the preconditioned gradient (Search Direction)
        preconditioned_gradient_flat = state['L_inv_root'] @ G @ state['R_inv_root']
        
        # 4. Update parameters
        if num_dims == 2:
            # Update Dense Layer
            layer.weights += -self.learning_rate * preconditioned_gradient_flat
        elif num_dims == 4:
            # Reshape back and Update Conv Layer
            
            # The weights were flattened via transpose(3, 0, 1, 2).reshape(m, n)
            # We need to reshape the update back: (m, n) -> (C_out, k_h, k_w, C_in) -> (k_h, k_w, C_in, C_out)
            update_4d = preconditioned_gradient_flat.reshape(original_shape[3], *original_shape[:3]).transpose(1, 2, 3, 0)
            layer.weights += -self.learning_rate * update_4d

        # Biases (simple SGD update, as is common with Shampoo)
        if layer.bias is not None:
             layer.bias += -self.learning_rate * layer.dbiases
