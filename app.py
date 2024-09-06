from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

#load the trained model (assuming 'model.joblib' is your Iris dataset model)
model = joblib.load('model.joblib')

#route to display the homepage with the form
@app.route('/')
def index():
    return render_template('index.html')

#route to process user inputs and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        #get form data from the request
        sepal_length = float(request.form['sepal_length'])
        sepal_width = float(request.form['sepal_width'])
        petal_length = float(request.form['petal_length'])
        petal_width = float(request.form['petal_width'])
        
        #create the input array for the model
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        #make prediction
        prediction = model.predict(input_features)
        
        #map the numeric prediction to the corresponding Iris species
        iris_classes = ['Setosa', 'Versicolor', 'Virginica']
        predicted_class = iris_classes[int(prediction[0])]

        #render the result back to the user
        return render_template('index.html', prediction_text=f'The predicted Iris species is: {predicted_class}')
    
    except Exception as e:
        return f"Error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
