import numpy as np

class Model():
    def __init__(self, hidden_layers, loss, optimizer):
        self.hidden_layers = hidden_layers
        self.loss = loss
        self.optimizer = optimizer

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            for i, layer in enumerate(self.hidden_layers):
                if i == 0:
                    layer.forward(X)
                else:
                    layer.forward(self.hidden_layers[i-1].output)

            loss = self.loss.forward(layer.output, y)

            predictions = np.argmax(self.loss.output, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)

            accuracy = np.mean(predictions==y)
            if not epoch % 100:
                print("Epoch", epoch, "Accuracy", accuracy, "Loss", loss)

            self.loss.backward(self.loss.output, y)

            for i, layer in enumerate(reversed(self.hidden_layers)):
                if i == 0:
                    layer.backward(self.loss.dinputs)
                else:
                    layer.backward(list(reversed(self.hidden_layers))[i-1].dinputs)

            for layer in self.hidden_layers:
                if layer.trainable:
                    self.optimizer.update_params(layer)

    def test(self, data):
        for i, layer in enumerate(self.hidden_layers):
            if i == 0:
                layer.forward(data)
            else:
                layer.forward(self.hidden_layers[i-1].output)
        return layer.output