import numpy as np 

data = np.array([
    ['Hot', 'Good', 0],
    ['Hot', 'Bad', 0],
    ['Hot', 'Bad', 1],
    ['Cold', 'Bad', 1],
    ['Cold', 'Good', 1],
    ['Cold', 'Bad', 1],
    ['Hot', 'Bad', 0],
    ['Hot', 'Good', 1]
])

X = data[:, :-1]
y = data[:, -1].astype(int)

def entropyCalculate(y):
    values, counts = np.unique(y, return_counts=True)
    prob = counts / len(y)
    return -np.sum(prob * np.log2(prob))

def split(X_col, y, threshold):
    left_mask = (X_col == threshold)
    right_mask = (X_col != threshold)
    return y[left_mask], y[right_mask]

def information_gain(X_col, y, threshold):
    parent_entropy = entropyCalculate(y)
    y_l, y_r = split(X_col, y, threshold)
    if len(y_l) == 0 or len(y_r) == 0:
        return 0
    n = len(y)
    child_entropy = (len(y_l)/n)*entropyCalculate(y_l) + (len(y_r)/n)*entropyCalculate(y_r)
    return parent_entropy - child_entropy

best_gain = -1
best_feature = None
best_threshold = None

for i in range(X.shape[1]):
    values = np.unique(X[:,i])
    for t in values:
        gain = information_gain(X[:, i], y, t)
        if gain > best_gain:
            best_gain = gain
            best_feature = i
            best_threshold = t
print("Entropy of dataset:", entropyCalculate(y))
print("Best feature (root):", best_feature)
print("Best threshold:", best_threshold)
print("Max Information Gain:", best_gain)