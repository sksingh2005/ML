import math
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = []
        self.y_train = []

    # Store training data
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    # Euclidean distance
    def euclidean_distance(self, x1, x2):
        distance = 0
        for a, b in zip(x1, x2):
            distance += (a - b) ** 2
        return math.sqrt(distance)

    # Predict a single sample
    def predict_single(self, x):
        distances = []

        # Calculate distance from all training points
        for i in range(len(self.X_train)):
            distance = self.euclidean_distance(x, self.X_train[i])
            distances.append((distance, self.y_train[i]))

        # Sort by distance
        distances.sort(key=lambda item: item[0])

        # Get k nearest neighbors
        k_neighbors = distances[:self.k]

        # Extract labels
        labels = [label for _, label in k_neighbors]

        # Majority voting
        most_common = Counter(labels).most_common(1)
        return most_common[0][0]

    # Predict multiple samples
    def predict(self, X):
        predictions = []

        for x in X:
            prediction = self.predict_single(x)
            predictions.append(prediction)

        return predictions

X_train = [
    [1, 2],
    [2, 3],
    [3, 3],
    [6, 5],
    [7, 7],
    [8, 6]
]

y_train = ['A', 'A', 'A', 'B', 'B', 'B']

knn = KNN(k=3)

knn.fit(X_train, y_train)

X_test = [
    [2, 2],
    [7, 5]
]

predictions = knn.predict(X_test)

print("Predictions:", predictions)