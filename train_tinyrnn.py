import numpy as np
from layers import Recurrent
from optimizers import CustomSGD
from activation import SoftmaxLossCategoricalCrossEntropy

# Define some hyperparameters
input_size = 1
hidden_size = 32
output_size = 1

# Create two recurrent nodes using the Recurrent class
rnn1 = Recurrent(input_size, hidden_size, output_size)
rnn2 = Recurrent(output_size, hidden_size, output_size)
loss_activation = SoftmaxLossCategoricalCrossEntropy()

# Create an optimizer using the SGD class
optimizer = CustomSGD()

# Create a list with values of a time series
# For example: sin(x) + noise
x_list = []
y_list = []
for i in range(1000):
    x = i / 100.0
    noise = np.random.normal(0.0, 0.1)
    y = np.sin(x) + noise
    x_list.append(x)
    y_list.append(y)

x_list = np.random.rand(100).tolist()
y_list = np.random.rand(100).tolist()

# Convert the list to a numpy array of shape (1000, 1)
x_array = np.array(x_list).reshape(-1, 1)
y_array = np.array(y_list).reshape(-1, 1)

# Train the network for 100 epochs
for epoch in range(100000):
    # Initialize the total loss to zero
    total_loss = 0.0
    
    # Loop over the data points one by one
    for i in range(len(x_array)):
        # Get the input and output data point
        x = x_array[i] # shape (1,)
        y_true = y_array[i] # shape (1,)
        
        # Reshape them to have a batch dimension of 1
        x = x.reshape(1, 1) # shape (1, 1)
        y_true = y_true.reshape(1, 1) # shape (1, 1)
        
        # Forward pass through the first node
        rnn1.forward(x) # shape (1, 1)
        rnn2.forward(rnn1.output) # shape (1, 1)
        
        loss = loss_activation.forward(rnn2.output, y_true)
        predictions = np.argmax(loss_activation.output, axis=1)

        # Backward pass through the second node
        dW_xh2, dW_hh2, dW_hy2 = rnn2.backward(rnn1.output, rnn2.output, y_true)
        dW_xh1, dW_hh1, dW_hy1 = rnn1.backward(x, rnn1.output, rnn2.output)
        
        # Update the weights using the optimizer
        optimizer.update_params(rnn1.W_xh, dW_xh1)
        optimizer.update_params(rnn1.W_hh, dW_hh1)
        optimizer.update_params(rnn1.W_hy, dW_hy1)
        
        optimizer.update_params(rnn2.W_xh, dW_xh2)
        optimizer.update_params(rnn2.W_hh, dW_hh2)
        optimizer.update_params(rnn2.W_hy, dW_hy2)

    if len(y.shape)  == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions==y)

    if epoch % 1000:
        print("Epoch", epoch*1000, "Accuracy", accuracy, "Loss", loss)#, "Gradients", dW_xh1, dW_hh1, dW_hy1)