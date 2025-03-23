# Molecule Solubility Prediction

## Overview
This project demonstrates a machine learning approach to predict the solubility of molecules using molecular descriptors. The dataset used contains molecular properties such as LogP, molecular weight, number of rotatable bonds, and aromatic proportion to predict the solubility (logS) of molecules.

## Dataset
The dataset used in this project is **Delaney's solubility dataset** with molecular descriptors, sourced from:
```
https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv
```
### Features:
- **MolLogP**: Log partition coefficient (lipophilicity)
- **MolWt**: Molecular weight
- **NumRotatableBonds**: Number of rotatable bonds
- **AromaticProportion**: Proportion of aromatic atoms
- **logS (Target Variable)**: Experimental solubility

## Project Workflow
### 1️⃣ Data Preparation
- Load dataset using **pandas**
- Separate features (**X**) and target variable (**y**)
- Split data into **training** and **testing** sets (80-20 split)

```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")

# Separate features and target variable
y = df['logS']
x = df.drop('logS', axis=1)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
```

### 2️⃣ Model Training & Evaluation
We train and compare two regression models:
- **Linear Regression**
- **Random Forest Regressor**

#### **Linear Regression**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lr = LinearRegression()
lr.fit(x_train, y_train)

# Predictions
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

# Performance Metrics
lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)
lr_train_r2 = r2_score(y_train, y_lr_train_pred)

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)
lr_test_r2 = r2_score(y_test, y_lr_test_pred)
```

#### **Random Forest Regressor**
```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)
rf.fit(x_train, y_train)

# Predictions
y_rf_train_pred = rf.predict(x_train)
y_rf_test_pred = rf.predict(x_test)

# Performance Metrics
rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)
rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)
rf_test_r2 = r2_score(y_test, y_rf_test_pred)
```

### 3️⃣ Model Comparison
Performance comparison between **Linear Regression** and **Random Forest**:

| Model | Training MSE | Training R2 | Test MSE | Test R2 |
|--------|-------------|------------|----------|---------|
| Linear Regression | 1.0075 | 0.7645 | 1.0207 | 0.7892 |
| Random Forest | 1.0282 | 0.7597 | 1.4077 | 0.7092 |

### 4️⃣ Data Visualization
A scatter plot visualizing **predicted vs actual logS values** for Linear Regression.

```python
import matplotlib.pyplot as plt
import numpy as np

plt.scatter(y_train, y_lr_train_pred, c="green", alpha=0.3)

z = np.polyfit(y_train, y_lr_train_pred, 1)
p = np.poly1d(z)
plt.plot(y_train, p(y_train), "red")

plt.xlabel("Experimental LogS")
plt.ylabel("Predicted LogS")
plt.title("Linear Regression: Experimental vs Predicted logS")
plt.show()
```

## Conclusion
- **Linear Regression** performed better than **Random Forest Regressor** based on R² and MSE values.
- The model provides a reasonable estimate of **molecular solubility (logS)** based on molecular descriptors.
- Further improvements can be achieved with **feature engineering, hyperparameter tuning, and additional descriptors**.

## Requirements
To run this project, install the necessary dependencies:
```bash
pip install pandas scikit-learn matplotlib numpy
```

## Author
**[Your Name]** - *Data Science Enthusiast*

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
✅ **Feel free to contribute, improve, or ask questions!** 🚀

