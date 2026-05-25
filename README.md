
# 🚗 Car Price Prediction System


## 📌 What is This Project?

This project is a **Car Price Prediction System** — a web application built using **Machine Learning** and **Flask**.

When a user visits the website, they fill in details about a car like:
- Which brand it is (Tesla, BMW, Toyota, etc.)
- What year it was made
- How big the engine is
- Whether it runs on Petrol or Diesel
- Whether it has Manual or Automatic transmission
- How many kilometres it has been driven (Mileage)
- What condition it is in (New / Used / Like New)
- What model it is (Model X, Mustang, Innova, etc.)

After filling all these details and clicking **Predict Price**, the system uses a trained Machine Learning model to calculate and display the **estimated price of the car**.

This is very useful for:
- A **buyer** who wants to know if a car price is fair
- A **seller** who wants to set a reasonable price
- A **dealer** who wants to quickly estimate car values

---

## 🎯 Why Did I Build This Project?

I built this project because:
- I wanted to learn how Machine Learning works on real data
- I wanted to connect a Python ML model with a website using Flask
- I wanted to understand how different car features like year, mileage, and brand affect the final price
- I wanted to build a complete project from collecting data, training a model, building a website, and deploying it live
- This project helps me grow as a **Data Analyst and ML Engineer**

---

## 🛠️ Tools and Technologies Used

| Tool / Technology | What It Does In This Project |
|---|---|
| **Python** | Main programming language used for everything |
| **Pandas** | Used to load the dataset and clean the data |
| **NumPy** | Used for numerical operations and array handling |
| **Scikit-learn** | Used to build and train the Machine Learning model |
| **Matplotlib** | Used to draw graphs and charts during analysis |
| **Seaborn** | Used to make the graphs look better and colorful |
| **Flask** | Used to build the website that runs the ML model |
| **HTML & CSS** | Used to design the front page of the website |
| **Jinja2** | Used to pass ML results from Python into the HTML page |
| **Joblib** | Used to save the trained model and load it later |
| **GitHub** | Used to store and share the project code |
| **Render** | Used to deploy the project live on the internet |
| **VS Code** | Code editor used to write all the code |

---

## 📊 Dataset — What Data Did I Use?

The dataset contains information about different cars. Each row represents one car with these columns:

| Column Name | What It Means | Example |
|---|---|---|
| **Brand** | The company that made the car | Tesla, BMW, Toyota |
| **Year** | The year the car was manufactured | 2018, 2020, 2022 |
| **Engine Size** | Size of the car engine in litres | 1.5, 2.0, 3.5 |
| **Fuel Type** | Type of fuel the car uses | Petrol, Diesel |
| **Transmission** | How the car changes gears | Manual, Automatic |
| **Mileage** | Total kilometres driven so far | 15000, 45000, 90000 |
| **Condition** | Current state of the car | New, Used, Like New |
| **Model** | Specific model name of the car | Model X, Innova, Mustang |
| **Price** | The actual price of the car — this is what we predict | ₹ 5,00,000 |

---

## 🧮 Manual Calculations (Step by Step)

### 📍 Step 1 — Label Encoding (Converting Text to Numbers)

Machine Learning models cannot understand text like "BMW" or "Petrol". So we convert all text values into numbers. This is called **Label Encoding**.

| Brand | Number |
|---|---|
| Tesla | 0 |
| BMW | 1 |
| Audi | 2 |
| Ford | 3 |
| Honda | 4 |
| Mercedes | 5 |
| Toyota | 6 |

| Fuel Type | Number |
|---|---|
| Petrol | 0 |
| Diesel | 1 |

| Transmission | Number |
|---|---|
| Manual | 0 |
| Automatic | 1 |

| Condition | Number |
|---|---|
| New | 0 |
| Used | 1 |
| Like New | 2 |

| Model | Number |
|---|---|
| Model X | 0 |
| 5 Series | 1 |
| A4 | 2 |
| Mustang | 3 |
| City | 4 |
| C Class | 5 |
| Innova | 6 |

---

### 📍 Step 2 — Multiple Linear Regression Formula

Linear Regression is a method where the model learns a mathematical formula to predict price based on input values.

The formula is:

Price = b0 + b1×Brand + b2×Year + b3×Engine + b4×Fuel + b5×Transmission + b6×Mileage + b7×Condition + b8×Model

What each part means:
- b0 = Base price (starting value before any features)
- b1 to b8 = Coefficients — weights the model learned from training data
- Each coefficient tells how much that feature increases or decreases the price
- Example: if b2 = 200 for Year — every 1 year newer adds ₹200 to the price
- Example: if b6 = -0.05 for Mileage — every 1 km driven reduces price by ₹0.05

---

### 📍 Step 3 — How the Model Trains

During training:
1. The model sees hundreds of rows of car data
2. It tries to find the best values for b0, b1, b2 ... b8
3. It checks how close its predictions are to actual prices
4. It adjusts the coefficients again and again until predictions are accurate
5. The error is measured using Mean Squared Error (MSE):

MSE = (1/n) × Σ (Actual Price - Predicted Price)²

The model tries to make this MSE as small as possible.

---

### 📍 Step 4 — Manual Calculation Example

Input from user:

Brand        = BMW        → encoded = 1
Year         = 2020
Engine Size  = 2.0
Fuel Type    = Diesel     → encoded = 1
Transmission = Automatic  → encoded = 1
Mileage      = 30000
Condition    = Used       → encoded = 1
Model        = 5 Series   → encoded = 1

Model coefficients (example values):

b0  =  5000   (base price)
b1  =  3000   (Brand)
b2  =  200    (Year)
b3  =  1500   (Engine Size)
b4  =  500    (Fuel Type)
b5  =  1000   (Transmission)
b6  = -0.05   (Mileage — negative means more km = lower price)
b7  = -2000   (Condition — Used = lower price)
b8  =  1200   (Model)

Applying the formula:

Price = 5000
      + (3000  × 1)      =  3000    → BMW brand
      + (200   × 2020)   =  404000  → Year 2020
      + (1500  × 2.0)    =  3000    → Engine 2.0L
      + (500   × 1)      =  500     → Diesel
      + (1000  × 1)      =  1000    → Automatic
      + (-0.05 × 30000)  = -1500    → 30000 km driven
      + (-2000 × 1)      = -2000    → Used condition
      + (1200  × 1)      =  1200    → 5 Series model

Price = 5000 + 3000 + 404000 + 3000 + 500 + 1000 - 1500 - 2000 + 1200

Price = ₹ 4,15,200 (approximately)

> ✅ In real code, scikit-learn automatically finds the best coefficients. We do not set them manually.

---

### 📍 Step 5 — Train Test Split

Before training we split the dataset into two parts:

Total Dataset  →  100%
Training Data  →   80%  ← Model learns from this
Testing Data   →   20%  ← Model is tested on unseen data

Why do we split?
- If we train and test on the same data the model will just memorize it
- Testing on unseen data tells us how well the model works in real life
- This is called checking for Overfitting

---

### 📍 Step 6 — Model Accuracy Metrics

R² Score:
R² = 1 - (SS_residual / SS_total)
Value between 0 and 1. Closer to 1 means the model is very accurate.
Example: R² = 0.92 means model explains 92% of price variation.

Mean Absolute Error (MAE):
MAE = Average of |Actual Price - Predicted Price|
Average difference between actual and predicted price. Lower is better.

Mean Squared Error (MSE):
MSE = Average of (Actual Price - Predicted Price)²
Penalizes large errors more. Lower is better.

Root Mean Squared Error (RMSE):
RMSE = √MSE
Same unit as price so easier to understand. Lower is better.

---

## 📈 Visualizations — Graphs Plotted

### 1. 📊 Brand vs Average Price (Bar Chart)
- X-axis: Car Brand
- Y-axis: Average Price
- Finding: Tesla and Mercedes have the highest average prices. Honda and Toyota are more affordable.

### 2. 📉 Mileage vs Price (Scatter Plot)
- X-axis: Mileage (km driven)
- Y-axis: Car Price
- Finding: As mileage increases price goes down — negative correlation. More km driven means more wear and tear so lower price.

### 3. 📈 Year vs Price (Scatter Plot)
- X-axis: Year of Manufacture
- Y-axis: Car Price
- Finding: Newer cars have higher prices — positive correlation.

### 4. 🥧 Fuel Type Distribution (Pie Chart)
- Shows what percentage of cars use Petrol vs Diesel
- Finding: Most cars in the dataset are Petrol-powered.

### 5. 🌡️ Correlation Heatmap
- Shows how strongly each feature is connected to Price
- Finding: Year and Engine Size have the strongest positive connection with Price. Mileage has a negative connection.

### 6. 🎯 Actual vs Predicted Price (Scatter Plot)
- X-axis: Actual Price
- Y-axis: Predicted Price by Model
- Finding: Most points are close to the diagonal line which means model predictions are very close to real prices.

---

## 🚀 How to Run This Project on Your Computer

**Step 1 — Download the project**
git clone https://github.com/yourusername/car-price-prediction.git
cd car-price-prediction

**Step 2 — Install all required libraries**
pip install -r requirements.txt

**Step 3 — Train the Machine Learning model**
python model.py
This creates a file called car_model.pkl which is the saved trained model.

**Step 4 — Start the Flask web application**
python app.py

**Step 5 — Open the website in your browser**
http://127.0.0.1:5000
You will see the Car Price Prediction form. Fill in the details and click Predict Price!

---

## 📁 Project Folder Structure

car-price-prediction/
│
├── static/                        ← All images used in the website
│   ├── vt image.jpeg              ← Vihara Tech company logo
│   ├── Tesla-Motors-Model-S.jpg   ← Tesla car image shown in output
│   ├── bmw-7-series-1.webp        ← BMW car image shown in output
│   ├── audi-a4-1.webp             ← Audi car image shown in output
│   ├── ford.avif                  ← Ford car image shown in output
│   ├── honda.webp                 ← Honda car image shown in output
│   ├── benz.avif                  ← Mercedes car image shown in output
│   └── toyota.jpg                 ← Toyota car image shown in output
│
├── templates/
│   └── index.html                 ← The main webpage with form and output
│
├── app.py                         ← Flask app — connects website with ML model
├── model.py                       ← Code to train and save the ML model
├── car_model.pkl                  ← Saved trained ML model file
├── requirements.txt               ← List of all Python libraries needed
├── Procfile                       ← Tells Render how to run the app
└── README.md                      ← This file you are reading now

---

## 📦 Libraries to Install

flask
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
gunicorn

Install all at once:
pip install -r requirements.txt

---

## 🌐 How is This Project Deployed?

This project is hosted live on the internet using Render — a free cloud platform.

Steps followed to deploy:
1. Pushed all code to GitHub
2. Connected GitHub repository to Render
3. Render reads the Procfile to know how to start the app
4. The app runs live and anyone can access it from anywhere

Deployment files used:
- Procfile → web: gunicorn app:app
- requirements.txt → installs all libraries on Render server
- runtime.txt → tells Render which Python version to use

---

## 🤝 Developed By

**KAMMARI RAMYA**

💼 Role — Data Analyst and ML Engineer

🏢 Training — Vihara Tech Private Limited

📧 Email — kammariramyachary@gmail.com

🌐 Live Demo — [Click Here to View the Application](https://car-price-prediction-6ysr.onrender.com)

---
