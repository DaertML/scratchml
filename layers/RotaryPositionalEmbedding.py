import numpy as np

class RotaryPositionalEmbedding:
    def __init__(self, d_model, max_position=2048):
        """
        Initialize Rotary Positional Embedding layer

        Args:
            d_model: Dimension of the model (embeddings)
            max_position: Maximum sequence length supported
        """
        self.d_model = d_model
        self.max_position = max_position

        # Ensure d_model is even for RoPE implementation
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for rotary positional embeddings")

        self.d_k = d_model // 2  # Dimension of each head

        # Pre-compute position embeddings for all positions
        self._compute_freqs()

    def _compute_freqs(self):
        """
        Compute frequency embeddings for RoPE
        """
        # Compute frequencies for different dimensions
        # Following the Llama paper: freqs = 10000^(-2*i/d_model) where i is dimension index
        self.freqs = np.power(10000, -2 * np.arange(0, self.d_k, dtype=np.float32) / self.d_k)

        # Pre-compute sine and cosine tables for efficiency
        self._sin_cache = {}
        self._cos_cache = {}

    def _get_freqs_for_position(self, position):
        """
        Get frequency embeddings for a specific position

        Args:
            position: Position in sequence

        Returns:
            Frequency embeddings for the position
        """
        if position >= self.max_position:
            # Handle positions beyond max_position - use scaled frequencies
            freqs = self.freqs * np.power(position / self.max_position, 2 * self.d_k / self.d_model)
        else:
            freqs = self.freqs

        return freqs

    def _apply_rope(self, x, position):
        """
        Apply rotary positional embedding to a tensor

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model) or (batch_size, num_heads, seq_len, d_k)
            position: Position in sequence (int or array of positions)

        Returns:
            Tensor with rotary positional embeddings applied
        """
        # For simplicity and compatibility, we'll assume x has shape (..., d_model)
        batch_shape = x.shape[:-1]
        d_model = x.shape[-1]

        if d_model != self.d_model:
            raise ValueError(f"Expected d_model {self.d_model}, got {d_model}")

        # Split into two halves for rotation
        x1 = x[..., :self.d_k]  # First half
        x2 = x[..., self.d_k:]  # Second half

        # Get frequencies for this position
        if isinstance(position, int):
            freqs = self._get_freqs_for_position(position)
            freqs = freqs.reshape(1, -1)  # Shape: (1, d_k)
        else:
            # Handle array of positions
            freqs = np.array([self._get_freqs_for_position(pos) for pos in position])

        # Apply rotation - this is the core RoPE operation
        # For each dimension i: x_rotated[i] = x[i] * cos(freq) - x[i+d_k] * sin(freq)
        # and x_rotated[i+d_k] = x[i] * sin(freq) + x[i+d_k] * cos(freq)

        # Compute sine and cosine values
        if isinstance(position, int):
            if position not in self._sin_cache:
                self._sin_cache[position] = np.sin(freqs)
                self._cos_cache[position] = np.cos(freqs)
            sin_vals = self._sin_cache[position]
            cos_vals = self._cos_cache[position]
        else:
            # For multiple positions, compute directly
            sin_vals = np.sin(freqs)
            cos_vals = np.cos(freqs)

        # Apply RoPE rotation (simplified version for numpy)
        rotated_x1 = x1 * cos_vals - x2 * sin_vals
        rotated_x2 = x1 * sin_vals + x2 * cos_vals

        # Combine back
        rotated = np.concatenate([rotated_x1, rotated_x2], axis=-1)

        return rotated

    def forward(self, x, positions=None):
        """
        Forward pass of Rotary Positional Embedding

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            positions: Optional array of positions for each token.
                      If None, assumes sequential positions 0, 1, 2, ...

        Returns:
            Tensor with rotary positional embeddings applied
        """
        if positions is None:
            # Assume sequential positions
            seq_len = x.shape[1]
            positions = np.arange(seq_len)

        batch_size, seq_len, d_model = x.shape

        # Apply RoPE to each position
        result = np.zeros_like(x)
        for i in range(seq_len):
            pos = positions[i]
            result[:, i, :] = self._apply_rope(x[:, i:i+1, :], pos)

        return result

    def get_rotated_qk(self, q, k, positions=None):
        """
        Apply RoPE to query and key tensors directly for attention computation

        Args:
            q: Query tensor of shape (batch_size, num_heads, seq_len, d_k)
            k: Key tensor of shape (batch_size, num_heads, seq_len, d_k)
            positions: Optional array of positions for each token

        Returns:
            Tuple of rotated Q and K tensors
        """
        batch_size, num_heads, seq_len, d_k = q.shape

        if positions is None:
            # Assume sequential positions
            positions = np.arange(seq_len)

        # Apply RoPE to each position separately
        q_rotated = np.zeros_like(q)
        k_rotated = np.zeros_like(k)

        for i in range(seq_len):
            pos = positions[i]
            freqs = self._get_freqs_for_position(pos)
            freqs = freqs.reshape(1, -1)  # Shape: (1, d_k)

            # Compute sine and cosine
            sin_vals = np.sin(freqs)
            cos_vals = np.cos(freqs)

            # Apply rotation to query
            q_slice = q[:, :, i:i+1, :]
            q1 = q_slice[..., :d_k//2]  # First half
            q2 = q_slice[..., d_k//2:]  # Second half

            rotated_q1 = q1 * cos_vals - q2 * sin_vals
            rotated_q2 = q1 * sin_vals + q2 * cos_vals

            q_rotated[:, :, i:i+1, :] = np.concatenate([rotated_q1, rotated_q2], axis=-1)

            # Apply rotation to key
            k_slice = k[:, :, i:i+1, :]
            k1 = k_slice[..., :d_k//2]  # First half
            k2 = k_slice[..., d_k//2:]  # Second half

            rotated_k1 = k1 * cos_vals - k2 * sin_vals
            rotated_k2 = k1 * sin_vals + k2 * cos_vals

            k_rotated[:, :, i:i+1, :] = np.concatenate([rotated_k1, rotated_k2], axis=-1)

        return q_rotated, k_rotated

    def get_params(self):
        """Return current parameters for optimizer updates"""
        # RoPE has no learnable parameters
        return []

    def set_params(self, params):
        """Set new parameter values (no-op for RoPE)"""
        pass

    def reset_cache(self):
        """Reset any cached sine/cosine values"""
        self._sin_cache.clear()
        self._cos_cache.clear()