# 📊 AI Customer Churn Prediction System

An end-to-end Deep Learning (ANN) project for predicting customer churn using TensorFlow/Keras and deployed with Streamlit.



## 🚀 Overview

Customer churn prediction helps businesses identify customers who are likely to leave.

This project uses an Artificial Neural Network (ANN) trained on structured banking data to predict churn probability and support customer retention strategies.



## 🎯 Problem Statement

Predict whether a bank customer will:

- Exit (Churn = 1)
- Stay (Churn = 0)

Using features like credit score, geography, age, balance, and activity status.



## 🧠 Machine Learning Pipeline

Raw Data (CSV)  
→ Data Cleaning  
→ Encoding (Label + OneHot)  
→ Feature Scaling (StandardScaler)  
→ Train-Test Split  
→ ANN Model Training  
→ Evaluation  
→ Saved Model  
→ Streamlit Deployment  



## 📦 Dataset Information

- Dataset: Churn Modelling Dataset  
- Records: 10,000  
- Target: Exited  
- Type: Binary Classification  



## 🧹 Data Preprocessing

### Removed Columns
- RowNumber
- CustomerId
- Surname

### Encoding
- Gender → Label Encoding
- Geography → One-Hot Encoding (drop first)

### Scaling
StandardScaler used for feature normalization.



## 🧠 Model Architecture

```python
model = Sequential([
    Dense(11, activation='relu', input_dim=11),
    Dense(7, activation='relu'),
    Dense(1, activation='sigmoid')
])