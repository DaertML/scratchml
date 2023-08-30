import numpy as np

class MaxPool:

    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1
        output = np.zeros((batch_size, in_channels, out_height, out_width))
        self.mask = np.zeros_like(input)

        for b in range(batch_size):
            for c in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        input_region = input[b, c, i * self.stride : i * self.stride + self.pool_size, j * self.stride : j * self.stride + self.pool_size]

                        output[b, c, i, j] = np.max(input_region)

                        max_index = np.argmax(input_region)
                        max_i = i * self.stride + max_index // self.pool_size
                        max_j = j * self.stride + max_index % self.pool_size
                        self.mask[b, c, max_i, max_j] = 1

        self.output = output
        return output
    
    def backward(self, dinputs):
        batch_size, out_channels, out_height, out_width = dinputs.shape
        input_gradient = np.zeros_like(self.mask)

        for b in range(batch_size):
            for c in range(out_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        input_gradient_region = input_gradient[b, c, i * self.stride : i * self.stride + self.pool_size, j * self.stride : j * self.stride + self.pool_size]

                        max_index = np.argmax(self.mask[b, c, i * self.stride : i * self.stride + self.pool_size, j * self.stride : j * self.stride + self.pool_size])

                        input_gradient_region.flat[max_index] = dinputs[b, c, i, j]

        self.dinputs = input_gradient
        return input_gradient