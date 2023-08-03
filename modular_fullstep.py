from activation import Relu
from layers import Dense
from activation import SoftmaxLossCategoricalCrossEntropy
from optimizers import SGD

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()


X, y = spiral_data(samples=100, classes=3)

dense1 = Dense(2, 64)
activation1 = Relu()
dense2 = Dense(64, 3)
loss_activation = SoftmaxLossCategoricalCrossEntropy()
optimizer = SGD()

for epoch in range(10000):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape)  == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(predictions==y)
    if not epoch % 100:
        print("Epoch", epoch, "Accuracy", accuracy, "Loss", loss)

    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)

