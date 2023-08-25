import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv("shrink_ray_dataset.csv")
X = data["Power"].values.reshape(-1, 1)
y = data["Shrinkage"].values.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Plot the training data, test data, and regression line
plt.scatter(X_train, y_train, label='Power')
plt.scatter(X_test, y_test, label='Shrinkage')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
target_shrinkage = 85
predicted_knob_value = (target_shrinkage - model.intercept_) / model.coef_
print(f"Predicted Knob Value for Target Shrinkage 0.5: {predicted_knob_value[0][0]}")

