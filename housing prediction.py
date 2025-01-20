#Lets create model for Boston Housing Price Prediction
import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('./dataset/HousingData.csv')

# Check for missing values
print(df.isnull().sum())
# Impute missing values for numerical columns with the median
df.fillna(df.median(), inplace=True)

from sklearn.preprocessing import MinMaxScaler

# Separate features (X) and target variable (y)
X = df.drop(columns=['MEDV'])
y = df['MEDV']

# Normalize the features using MinMaxScaler
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

# Predict on test data
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
