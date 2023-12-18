from activation import Relu
from layers import Dense
from models import Model
from activation import SoftmaxLossCategoricalCrossEntropy
from optimizers import SGD

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

X, y = spiral_data(samples=100, classes=3)

hidden_layers = [
    Dense(2, 64),
    Relu(),
    Dense(64, 3)
]

loss_activation = SoftmaxLossCategoricalCrossEntropy()
optimizer = SGD()

model = Model(hidden_layers=hidden_layers, loss=loss_activation, optimizer=optimizer)
model.train(X, y, 1000)

out = model.test(X[0])
print(out)

