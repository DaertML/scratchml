import numpy as np
from layers import Conv2D, MaxPool
from activation import SoftmaxLossCategoricalCrossEntropy, Relu
from optimizers import SOAP

conv1 = Conv2D(1, 3, 3, 1, 0)
maxpool1 = MaxPool(2, 2)
loss_activation = SoftmaxLossCategoricalCrossEntropy()
optimizer = SOAP()

X = np.random.randn(2, 1, 4, 4)
y = np.array([1,0])
print(X)
for epoch in range(10000):
    conv1.forward(X)
    maxpool1.forward(conv1.output)

    loss = loss_activation.forward(maxpool1.output, y)
    predictions = np.argmax(loss_activation.output, axis=1)

    if len(y.shape)  == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if not epoch % 100:
        print("Epoch", epoch, "Accuracy", accuracy, "Loss", loss)

    loss_activation.backward(loss_activation.output, y)
    maxpool1.backward(loss_activation.dinputs)
    conv1.backward(maxpool1.dinputs)

    optimizer.update_params(conv1)
