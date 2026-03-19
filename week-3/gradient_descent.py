import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

X = (X - X.mean()) / X.std()

m = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

for i in range(epochs):
    y_pred = m * X + b
    dm = (-2/n) * np.sum(X * (y - y_pred))
    db = (-2/n) * np.sum(y - y_pred)
    m = m - learning_rate * dm
    b = b - learning_rate * db

print(m, b)

plt.scatter(X, y)
plt.plot(X, m * X + b)
plt.show()
