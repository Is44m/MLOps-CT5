import numpy as np
import pandas as pd

# Load the dataset
data = pd.read_csv('Iris.csv')

# Check the dataset's structure
print(data.head())

# Map species to binary classes (e.g., 0 and 1) for simplicity
data['Species'] = data['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Filter only class 0 and class 1
data = data[data['Species'] < 2]

# Features and labels
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values
y = data['Species'].values

# Split the dataset manually
np.random.seed(42)
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split_index = int(0.8 * X.shape[0])
train_indices, test_indices = indices[:split_index], indices[split_index:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Add intercept term to features
X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])

# Initialize weights
def initialize_weights(n_features):
    return np.zeros(n_features)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function (cross-entropy)
def compute_cost(X, y, weights):
    m = len(y)
    if m == 0:
        raise ValueError("Number of samples (m) should not be zero.")
    predictions = sigmoid(np.dot(X, weights))
    cost = (-1 / m) * (np.dot(y, np.log(predictions + 1e-10)) + np.dot(1 - y, np.log(1 - predictions + 1e-10)))
    return cost

# Gradient descent
def gradient_descent(X, y, weights, learning_rate, num_iterations):
    m = len(y)
    if m == 0:
        raise ValueError("Number of samples (m) should not be zero.")
    for _ in range(num_iterations):
        predictions = sigmoid(np.dot(X, weights))
        gradient = (1 / m) * np.dot(X.T, (predictions - y))
        weights -= learning_rate * gradient
    return weights

# Train logistic regression model
def train_logistic_regression(X_train, y_train, learning_rate=0.01, num_iterations=1000):
    weights = initialize_weights(X_train.shape[1])
    weights = gradient_descent(X_train, y_train, weights, learning_rate, num_iterations)
    return weights

# Train the model
weights = train_logistic_regression(X_train, y_train)

# Predict function
def predict(X, weights):
    predictions = sigmoid(np.dot(X, weights))
    return predictions >= 0.5

# Evaluate the model
def evaluate_model(X_test, y_test, weights):
    predictions = predict(X_test, weights)
    accuracy = np.mean(predictions == y_test)
    return accuracy

# Test the model
accuracy = evaluate_model(X_test, y_test, weights)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Save the model weights
np.save('model_weights.npy', weights)
