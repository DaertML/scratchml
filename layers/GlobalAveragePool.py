import numpy as np

class GlobalAveragePool:

    def __init__(self):
        pass

    def forward(self, input):
        """
        Forward pass for Global Average Pooling.
        Reduces spatial dimensions (H x W) to 1 x 1 for each channel.

        Args:
            input: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            output: Pooled tensor of shape (batch_size, channels, 1, 1)
        """
        self.input = input
        batch_size, channels, height, width = input.shape

        # Compute global average pooling across spatial dimensions
        output = np.mean(input, axis=(2, 3), keepdims=True)

        self.output = output
        return output

    def backward(self, gradient):
        """
        Backward pass for Global Average Pooling.

        Args:
            gradient: Gradient tensor of shape (batch_size, channels, 1, 1)

        Returns:
            input_gradient: Gradient tensor of shape (batch_size, channels, height, width)
        """
        batch_size, channels, _, _ = gradient.shape

        # Broadcast the gradient to match input dimensions
        # Each spatial location gets the same gradient value from the pooled output
        input_gradient = np.full_like(self.input, gradient / (self.input.shape[2] * self.input.shape[3]))

        self.dinputs = input_gradient
        return input_gradient