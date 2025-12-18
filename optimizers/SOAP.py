import numpy as np
from numpy.linalg import eigh, LinAlgError

# --- Helper Function for Eigen-decomposition (SOAP requirement) ---
def _get_eigenbasis(matrix):
    """
    Computes the eigenvectors (Q) from the preconditioner matrix.
    
    Args:
        matrix (np.ndarray): The input symmetric matrix (L or R preconditioner).
        
    Returns:
        np.ndarray: The matrix Q containing the eigenvectors.
    """
    # Use eigh for symmetric/Hermitian matrices (L and R should be symmetric PSD)
    try:
        # eigenvalues (s), eigenvectors (Q)
        s, Q = eigh(matrix)
        # Ensure eigenvalues are non-negative (clipping to 0 or epsilon might be needed in practice)
        # Sort eigenvectors by eigenvalue magnitude if necessary, but eigh usually sorts them.
        return Q
    except LinAlgError as e:
        print(f"LinAlgError in _get_eigenbasis: {e}")
        # Return identity matrix for safe fallback
        return np.eye(matrix.shape[0])


# --- SOAP Optimizer Class ---
class SOAP:
    def __init__(self, learning_rate=1e-3, beta1=0.9, beta2=0.999, epsilon=1e-6, precond_freq=10):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        # Frequency to compute the expensive eigen-decomposition
        self.precond_freq = precond_freq
        self.step_count = 0
        # State dictionary to store all per-layer variables
        self.state = {}

    def _get_or_init_state(self, layer, m, n):
        """Initializes or retrieves the state for a given layer."""
        if layer not in self.state:
            # Shampoo/SOAP Preconditioners (L and R)
            self.state[layer] = {
                'L': self.epsilon * np.eye(m),  # Left Preconditioner (m x m)
                'R': self.epsilon * np.eye(n),  # Right Preconditioner (n x n)
                'Q_L': np.eye(m),               # Left Eigenbasis Q_L (m x m)
                'Q_R': np.eye(n),               # Right Eigenbasis Q_R (n x n)
                # Adam State (Momentum and Variance - in coordinate space)
                'M': np.zeros((m, n)),          # First moment estimate
                'V': np.zeros((m, n))           # Second moment estimate (element-wise square)
            }
        return self.state[layer]

    def update_params(self, layer):
        self.step_count += 1
        t = self.step_count

        # --- 1. Handle Tensor Shape and Matricization (from previous fix) ---
        original_shape = layer.weights.shape
        num_dims = len(original_shape)
        
        if num_dims == 2:
            m, n = original_shape
            G_coord = layer.dweights
        elif num_dims == 4:
            # Conv weights shape: (k_h, k_w, C_in, C_out) -> Matricize to (C_out, k_h*k_w*C_in)
            m = original_shape[3]
            n = original_shape[0] * original_shape[1] * original_shape[2]
            G_coord = layer.dweights.transpose(3, 0, 1, 2).reshape(m, n)
        else:
            # Fallback to SGD for unsupported shapes
            layer.weights += -self.learning_rate * layer.dweights
            if layer.bias is not None:
                layer.bias += -self.learning_rate * layer.dbiases
            return

        state = self._get_or_init_state(layer, m, n)

        # --- 2. Update Shampoo Preconditioners (L and R) ---
        # L_t = L_{t-1} + G_t G_t^T
        state['L'] += G_coord @ G_coord.T
        # R_t = R_{t-1} + G_t^T G_t
        state['R'] += G_coord.T @ G_coord

        # --- 3. Compute Eigenbasis Periodically ---
        if t % self.precond_freq == 0:
            state['Q_L'] = _get_eigenbasis(state['L'])
            state['Q_R'] = _get_eigenbasis(state['R'])

        # --- 4. Rotate Gradient to Eigenbasis ---
        # G' = Q_L^T * G * Q_R
        Q_L_T = state['Q_L'].T
        G_rotated = Q_L_T @ G_coord @ state['Q_R']

        # --- 5. Adam Update in the Rotated Space ---
        
        # NOTE: The Shampoo state (L, R) is updated in the coordinate basis, 
        # but the Adam state (M, V) must be updated in the ROTATED basis for SOAP.
        
        # M' and V' are the Adam state *in the rotated space*
        M_rotated = state['M']
        V_rotated = state['V']
        
        # Update Adam moments in rotated space
        M_rotated = self.beta1 * M_rotated + (1 - self.beta1) * G_rotated
        V_rotated = self.beta2 * V_rotated + (1 - self.beta2) * (G_rotated**2) # Element-wise square
        
        # Bias correction (t is the global step count)
        M_hat = M_rotated / (1 - self.beta1**t)
        V_hat = V_rotated / (1 - self.beta2**t)

        # Update vector in rotated space (Adam-like update)
        # U_rotated = M_hat / (sqrt(V_hat) + epsilon)
        U_rotated = M_hat / (np.sqrt(V_hat) + self.epsilon)
        
        # --- 6. Rotate Update Vector back to Coordinate Space ---
        # U_coord = Q_L * U_rotated * Q_R^T
        U_coord = state['Q_L'] @ U_rotated @ state['Q_R'].T
        
        # Store the updated Adam moments (in the rotated space)
        state['M'] = M_rotated
        state['V'] = V_rotated

        # --- 7. Final Parameter Update ---
        if num_dims == 2:
            layer.weights += -self.learning_rate * U_coord
        elif num_dims == 4:
            # Reshape back and Update Conv Layer
            update_4d = U_coord.reshape(original_shape[3], *original_shape[:3]).transpose(1, 2, 3, 0)
            layer.weights += -self.learning_rate * update_4d

        # Biases: typically simple Adam or SGD (we use SGD for architecture matching)
        if layer.bias is not None:
             layer.bias += -self.learning_rate * layer.dbiases
