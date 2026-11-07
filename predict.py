# predict.py - Load the saved model and predict the price of a new house

import pandas as pd
import joblib

# Step 1: Load the trained model saved by train.py
model = joblib.load("model.pkl")

# Step 2: Describe the new house
# Column names must match the ones used in training: rooms, size_sqft
new_house = pd.DataFrame({"rooms": [3], "size_sqft": [1350]})

# Step 3: Ask the model to predict the price
# predict() returns a list, so [0] takes the first (and only) answer
predicted_price = model.predict(new_house)[0]

# Step 4: Print the result with 1 decimal place
print(f"3 rooms, 1350 sqft -> predicted price: {predicted_price:.1f} lakh")