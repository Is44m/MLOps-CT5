import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Binary classification setup: We will consider only two classes for simplicity
# Filter only class 0 and class 1
X = X[y < 2]
y = y[y < 2]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
    predictions = sigmoid(np.dot(X, weights))
    cost = (-1 / m) * (np.dot(y, np.log(predictions)) + np.dot(1 - y, np.log(1 - predictions)))
    return cost

# Gradient descent
def gradient_descent(X, y, weights, learning_rate, num_iterations):
    m = len(y)
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
joblib.dump(weights, 'model_weights.joblib')
