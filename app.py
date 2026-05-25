from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load Pickle Model
model = pickle.load(open("mini_project_2.pkl", "rb"))

# Brand and Model Mapping
brand_models = {
    "Tesla": ["Model S", "Model 3", "Model X", "Model Y"],
    "Audi": ["A4", "A6", "Q5", "Q7"],
    "BMW": ["X5", "3 Series", "5 Series"],
    "Ford": ["Mustang", "Explorer", "F-150"],
    "Honda": ["Civic", "Accord", "CR-V"],
    "Mercedes": ["C-Class", "E-Class", "GLA"],
    "Toyota": ["Corolla", "Camry", "Fortuner"]
}

# Encodings
brand_map = {
    "Tesla": 0,
    "Audi": 1,
    "BMW": 2,
    "Ford": 3,
    "Honda": 4,
    "Mercedes": 5,
    "Toyota": 6
}

fuel_map = {
    "Petrol": 0,
    "Diesel": 1,
    "Electric": 2,
    "Hybrid": 3
}

transmission_map = {
    "Manual": 0,
    "Automatic": 1
}

condition_map = {
    "New": 0,
    "Used": 1,
    "Like New": 2
}

# Model Encoding
model_map = {
    "Model S": 0,
    "Model 3": 1,
    "Model X": 2,
    "Model Y": 3,

    "A4": 4,
    "A6": 5,
    "Q5": 6,
    "Q7": 7,

    "X5": 8,
    "3 Series": 9,
    "5 Series": 10,

    "Mustang": 11,
    "Explorer": 12,
    "F-150": 13,

    "Civic": 14,
    "Accord": 15,
    "CR-V": 16,

    "C-Class": 17,
    "E-Class": 18,
    "GLA": 19,

    "Corolla": 20,
    "Camry": 21,
    "Fortuner": 22
}


@app.route("/")
def home():
    return render_template(
        "index.html",
        brand_models=brand_models
    )


@app.route("/predict", methods=["POST"])
def predict():

    brand = request.form["brand"]
    model_name = request.form["model"]
    year = int(request.form["year"])
    engine_size = float(request.form["engine_size"])
    fuel = request.form["fuel"]
    transmission = request.form["transmission"]
    mileage = int(request.form["mileage"])
    condition = request.form["condition"]

    # Encoding
    brand_encoded = brand_map[brand]
    model_encoded = model_map[model_name]
    fuel_encoded = fuel_map[fuel]
    transmission_encoded = transmission_map[transmission]
    condition_encoded = condition_map[condition]

    features = np.array([[
        brand_encoded,
        model_encoded,
        year,
        engine_size,
        fuel_encoded,
        transmission_encoded,
        mileage,
        condition_encoded
    ]])

    prediction = model.predict(features)[0]

    prediction = round(prediction, 2)

    return render_template(
        "index.html",
        prediction_text=f"Predicted Price : {prediction}",
        brand_models=brand_models
    )


if __name__ == "__main__":
    app.run(debug=True)