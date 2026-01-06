# ============================================================
# PHASE 5: MLP WITH BACKPROPAGATION (FROM SCRATCH)
# XOR CLASSIFICATION
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# XOR Dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

# Hyperparameters
learning_rate = 0.1
epochs = 5000

losses = []
accuracies = []

# Training Loop
for epoch in range(epochs):
    # Forward propagation
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)

    # Loss (MSE)
    loss = np.mean((y - y_hat) ** 2)
    losses.append(loss)

    # Accuracy
    predictions = (y_hat > 0.5).astype(int)
    accuracy = np.mean(predictions == y)
    accuracies.append(accuracy)

    # Backpropagation
    d_output = (y_hat - y) * sigmoid_derivative(y_hat)
    dW2 = np.dot(a1.T, d_output)
    db2 = np.sum(d_output, axis=0, keepdims=True)

    d_hidden = np.dot(d_output, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, d_hidden)
    db1 = np.sum(d_hidden, axis=0, keepdims=True)

    # Weight updates
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

# Plot Loss
plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Phase 5: Loss Reduction")
plt.show()

# Plot Accuracy
plt.plot(accuracies)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Phase 5: Accuracy Improvement")
plt.show()

print("Final XOR Accuracy:", accuracies[-1])
