import numpy as np

class Conv2D:

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size)
        self.bias = np.random.randn(out_channels)

    def forward(self, input):
        self.input = input
        batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        if self.padding > 0:
            input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant', constant_values=0)

        for b in range(batch_size):
            for c in range(self.out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        input_region = input[b, :, i * self.stride : i * self.stride + self.kernel_size, j * self.stride : j * self.stride + self.kernel_size]
                        output[b, c, i, j] = np.sum(input_region * self.weights[c]) + self.bias[c]

        self.output = output
        return output
    
    def backward(self, gradient):
        batch_size, out_channels, out_height, out_width = gradient.shape
        input_gradient = np.zeros_like(self.input)
        weights_gradient = np.zeros_like(self.weights)
        biases_gradient = np.zeros_like(self.bias)

        for b in range(batch_size):
            for c in range(out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        input_region = self.input[b, :, i * self.stride : i * self.stride + self.kernel_size, j * self.stride : j * self.stride + self.kernel_size]
                        weights_gradient[c] += gradient[b, c, i, j] * input_region
                        biases_gradient[c] += gradient[b, c, i, j]
                        input_gradient[b, :, i * self.stride : i * self.stride + self.kernel_size, j * self.stride : j * self.stride + self.kernel_size] += gradient[b, c, i, j] * np.flip(self.weights[c])

        self.dweights = weights_gradient
        self.dbiases = biases_gradient
        if self.padding > 0:
            return input_gradient[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            return input_gradient