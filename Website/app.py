# pip install flask
from datetime import datetime
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

# Loading the NumPy array model
model = joblib.load(open('Models/model.pkl', 'rb'))
# Flask is used for creating your application
# render template is used for rendering the HTML page
app = Flask(__name__)  # your application

@app.route('/')  # default route
def home():
    return render_template('index.html')  # rendering your home page.


@app.route("/pred", methods=['POST'])  # prediction route
def predict1():
    try:
        # Retrieve form data
        distance = float(request.form.get('distance'))
        source = request.form.get('sourceLocation')
        destination = request.form.get('destinationLocation')
        product_id = int(request.form.get('product_id'))
        name = request.form.get('name')

        # Create a NumPy array from the form data
        input_data = np.array([distance, source, destination, product_id, name])

        # Reshape the array to have a single row
        input_data = input_data.reshape(1, -1)

        # Print input data for debugging
        print("Input Array:")
        print(input_data)

        # Make prediction using the pre-trained model
        prediction = model.predict(input_data)
        print("Prediction:")
        print(prediction)

        return render_template("index.html", result="The predicted fare is " + str(np.round(prediction[0])))

    except Exception as e:
        print("Error:", e)
        return render_template("index.html", result="An error occurred. Please check the server logs for details.")

# running your application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
