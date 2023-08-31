class CustomSGD:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate

    def update_params(self, weights, gradients):
        weights += -self.learning_rate*gradients