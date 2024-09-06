from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

# Load the trained model weights (assuming 'model_weights.npy' is your file)
weights = np.load('model_weights.npy')

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the softmax function for multi-class classification
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0)

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
        
        # Create the input array for the model
        input_features = np.array([1, sepal_length, sepal_width, petal_length, petal_width])  # Adding 1 for intercept term
        
        # Make prediction using softmax for multi-class classification
        logits = np.dot(input_features, weights.T)  # weights.T for (num_classes x num_features)
        probabilities = softmax(logits)
        predicted_class = np.argmax(probabilities)
        
        # Map the numeric prediction to the corresponding Iris species
        iris_classes = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class_name = iris_classes[predicted_class]

        # Render the result back to the user
        return render_template('index.html', prediction_text=f'The predicted Iris species is: {predicted_class_name}')
    
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
