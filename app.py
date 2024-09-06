from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load the trained model weights (assuming 'model_weights.npy' is your file)
weights = np.load('model_weights.npy')

# Check the shape of weights to understand its dimensions
print("Weights shape:", weights.shape)  # For debugging

# Define the softmax function for multi-class classification
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Subtract max for numerical stability
    return exp_z / exp_z.sum(axis=1, keepdims=True)  # Perform softmax row-wise

# Route to display the homepage with the form
@app.route('/')
def index():
    return render_template('index.html')

# Route to process user inputs and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data from the request
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        # Create the input array for the model (Adding 1 for intercept term)
        input_features = np.array([1, sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        
        # Ensure input_features and weights dimensions match
        print("Input features shape:", input_features.shape)  # For debugging
        
        # Make prediction using softmax for multi-class classification
        logits = np.dot(input_features, weights.T)  # weights.T for (num_classes x num_features)
        probabilities = softmax(logits)
        predicted_class = np.argmax(probabilities, axis=1)[0]  # Get the predicted class
        
        # Map the numeric prediction to the corresponding Iris species
        iris_classes = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class_name = iris_classes[predicted_class]

        # Render the result back to the user
        return render_template('index.html', prediction_text=f'The predicted Iris species is: {predicted_class_name}')
    
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
