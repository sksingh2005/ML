import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Sample data
X = np.array([[1, 2],
              [2, 3],
              [3, 4],
              [4, 5]])

y = np.array([0, 0, 1, 1])

# Initialize weights
weights = np.zeros(X.shape[1])
bias = 0

learning_rate = 0.01
epochs = 1000

# Gradient Descent
for _ in range(epochs):
    linear = np.dot(X, weights) + bias
    y_pred = sigmoid(linear)

    dw = (1 / len(X)) * np.dot(X.T, (y_pred - y))
    db = (1 / len(X)) * np.sum(y_pred - y)

    weights -= learning_rate * dw
    bias -= learning_rate * db

# Prediction
test = np.array([[5, 6]])
prob = sigmoid(np.dot(test, weights) + bias)

print("Probability:", prob[0])
print("Class:", 1 if prob >= 0.5 else 0)