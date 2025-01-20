# #🏡 Boston Housing Price Prediction

This project demonstrates how to build a linear regression model to predict housing prices using the Boston Housing dataset.

## Prerequisites

- Python 3.x 🐍
- Jupyter Notebook or any Python IDE 🖥️

## Libraries

The following Python libraries are required:

- pandas 📊
- numpy 🔢
- scikit-learn 🛠️

You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn
```

## Dataset

The dataset used in this project is a CSV file containing housing data. Make sure to replace the path of the dataset in the code with the actual path to your CSV file:

```python
df = pd.read_csv('./dataset/HousingData.csv')
```

## Code Overview

### Importing Necessary Libraries

```python
import pandas as pd
import numpy as np
```

### Loading the Dataset
### Checking for Missing Values
### Imputing Missing Values
### Normalizing the Features
# Separate features (X) and target variable (y)
# Normalize the features using MinMaxScaler
### Splitting the Dataset
# Split data into training and testing sets (80% train, 20% test)
### Training the Model
![alt text](<Screenshot 2025-01-20 202930-1-1.png>)
### Making Predictions
# Predict on test data
### Evaluating the Model
![alt text](<Screenshot 2025-01-20 205805-1.png>)



## Results

The model's performance is evaluated using Root Mean Squared Error (RMSE), which measures the average magnitude of the errors between predicted and actual values.

- **Root Mean Squared Error (RMSE):** Indicates the standard deviation of the prediction errors.

## Conclusion

This project demonstrates a basic linear regression model to predict housing prices using the Boston Housing dataset. The model can be further improved by using more advanced algorithms, feature engineering, and hyperparameter tuning.

## Acknowledgements

- The dataset used in this project is sourced from the Boston Housing dataset.
- The project utilizes libraries like pandas for data manipulation and scikit-learn for machine learning.

Feel free to contribute to this project by opening issues or submitting pull requests. 🤝
```
