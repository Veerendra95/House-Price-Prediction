import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


print("This is my updated House Price Prediction Project!")
print("This is the line I have added in GitHub Repository")
print("This is the  second line I have added in GitHub Repository for testing")


# Creating a simple dataset
data = {
    "Area_sqft": [500, 1000, 1500, 2000, 2500, 3000, 3500],
    "Bedrooms": [1, 2, 2, 3, 3, 4, 4],
    "Location_Score": [3, 5, 7, 8, 9, 10, 10],
    "Price_in_lakhs": [20, 40, 60, 80, 100, 120, 140]
}

# Convert to DataFrame
df = pd.DataFrame(data)
print(df.head())


# Features (Independent Variables)
X = df[["Area_sqft", "Bedrooms", "Location_Score"]]

# Target (Dependent Variable)
y = df["Price_in_lakhs"]

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Check model coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")


# New house data
new_house = pd.DataFrame([[2800, 3, 8]],columns=["Area_sqft", "Bedrooms", "Location_Score"])
predicted_price = model.predict(new_house)

print(f"Predicted price for 2800 sqft, 3-bedroom house: ₹{predicted_price[0]:.2f} lakhs")


# Plot actual vs predicted prices
plt.scatter(range(len(y_test)), y_test, color="blue", label="Actual Price")
plt.scatter(range(len(y_pred)), y_pred, color="red", label="Predicted Price")
plt.xlabel("House Index")
plt.ylabel("Price in Lakhs")
plt.title("Actual vs Predicted House Prices")
plt.legend(loc="upper left")
plt.show()
