import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib

# Load the data
data_path = "TCS.csv"
data = pd.read_csv(data_path)

# Convert 'Date' to datetime type and set as index
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Handling missing values: fill missing 'Trades' with 0
data['Trades'] = data['Trades'].fillna(0)

# Check for other missing values and fill or drop
data.ffill(inplace=True)  # forward fill 

# Creating lag features for 'Close' to use previous day's close as a feature
data['Prev_Close'] = data['Close'].shift(1)

# Drop the first row as it now contains NaN (due to lag feature)
data.dropna(inplace=True)

# Define additional features
data['Prev_Volume'] = data['Volume'].shift(1)
data['SMA_5'] = data['Close'].rolling(window=5).mean().shift(1)
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean().shift(1)

#print("sma", data['Close'].rolling(window=5).mean())
#print("ema", data['Close'].ewm(span=10, adjust=False).mean())

#print('prev_close', data['Close'].shift(1))
#print('prev_volume', data['Volume'].shift(1))

# Drop any additional rows with NaN values created by rolling functions
data.dropna(inplace=True)

# Define the features (inputs) and the target (output)
features = ['Prev_Close', 'Prev_Volume', 'SMA_5', 'EMA_10']
target = 'Close'

# Select the features and target from the DataFrame
X = data[features]  # Features
y = data[target]    # Target

# Split the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

# Initialize the RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = rf_model.predict(X_test)

# Calculate the performance metrics, such as Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: {rmse}")

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Calculate R² Score
r2 = r2_score(y_test, y_pred)
print(f"R² Score: {r2}")

importances = rf_model.feature_importances_
feature_names = X_train.columns
feature_importance_dict = dict(zip(feature_names, importances))
print("Feature Importances:", feature_importance_dict)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Stock Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--')  # a reference line
plt.show()

# Save the model to disk
joblib.dump(rf_model, 'random_forest_stock_model.pkl')
