# train.py - Beginner ML lab: predict house prices with Linear Regression

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Step 1: Read the data from the CSV file
data = pd.read_csv("houses.csv")

# Step 2: Choose the features (X) and the label (y)
# X = the information we give the model, y = the answer we want it to learn
X = data[["rooms", "size_sqft"]]
y = data["price_lakh"]

# Step 3: Split the data into training and testing parts
# 80% for training, 20% for testing; random_state=42 keeps the split the same every run
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Houses used for training:", len(X_train))
print("Houses used for testing:", len(X_test))

# Step 4: Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict prices for the test houses and compare with the real prices
predictions = model.predict(X_test)

print("\nResults on test houses:")
for i in range(len(X_test)):
    rooms = X_test.iloc[i]["rooms"]
    size = X_test.iloc[i]["size_sqft"]
    real_price = y_test.iloc[i]
    guess = predictions[i]
    error = abs(real_price - guess)
    print(
        f"Rooms: {rooms}, Size: {size} sqft, "
        f"Real price: {real_price}, Guess: {guess:.1f}, Off by: {error:.1f}"
    )

# Step 6: Calculate the Mean Absolute Error (MAE)
# MAE = the average of how far off the guesses are
mae = mean_absolute_error(y_test, predictions)
print("\nMean Absolute Error (MAE):", round(mae, 2), "lakh")

# Step 7: Save the trained model to a file so we can use it later
joblib.dump(model, "model.pkl")
print("Model saved to model.pkl")