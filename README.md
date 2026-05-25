📊 AI Customer Churn Prediction System

An end-to-end Deep Learning (ANN) based Customer Churn Prediction System built using TensorFlow/Keras and deployed with Streamlit.
This system predicts whether a bank customer is likely to leave based on behavioral and financial attributes.

🚀 Project Overview

Customer churn is one of the most critical problems in the banking sector. Retaining existing customers is significantly cheaper than acquiring new ones.

This project uses a Artificial Neural Network (ANN) trained on structured customer data to predict churn probability and help businesses take proactive retention actions.

🧠 Problem Statement

Given customer details such as:

Credit Score
Geography
Gender
Age
Balance
Number of Products
Activity Status

👉 Predict whether the customer will Exit (Churn = 1) or Stay (Churn = 0).

🏗️ System Architecture


Raw Data (CSV)
      ↓
Data Preprocessing
  - Drop irrelevant columns
  - Label Encoding (Gender)
  - One-Hot Encoding (Geography)
  - Feature Scaling (StandardScaler)
      ↓
Train/Test Split
      ↓
Artificial Neural Network (Keras)
      ↓
Model Training (Class Weights for imbalance)
      ↓
Evaluation (ROC-AUC, Confusion Matrix, Recall)
      ↓
Saved Model (.keras)
      ↓
Streamlit Web App (Inference)


🧪 Dataset Information


Dataset: Churn Modelling Dataset
Rows: 10,000
Features: 13 input features + target
Target Variable: Exited
Feature Types:
Numerical: CreditScore, Age, Balance, Salary, etc.
Categorical: Geography, Gender


🧹 Data Preprocessing Pipeline


1. Feature Cleaning
Removed:
RowNumber
CustomerId
Surname
2. Encoding
Gender → Label Encoding
Geography → One-Hot Encoding (drop='first')
3. Scaling
StandardScaler applied to normalize features for ANN stability


🧠 Model Architecture (ANN)


model = Sequential([
    Dense(11, activation='relu', input_dim=11),
    Dense(7, activation='relu'),
    Dense(1, activation='sigmoid')
])
Activation Functions:
Hidden Layers → ReLU
Output Layer → Sigmoid (Binary Classification)
⚙️ Training Configuration
Loss Function: Binary Crossentropy
Optimizer: Adam
Metric: Recall
Epochs: 100
Class Weights:
{0: 1, 1: 2.5}

👉 Used to handle class imbalance (churn cases are fewer)

📈 Model Evaluation
🔹 ROC-AUC Score
0.8476
🔹 Confusion Matrix
[[1666  724]
 [ 111  499]]
🔹 Key Insight:
High recall achieved for churn class → important for business retention use-case


📊 Performance Visualization
Training vs Validation Loss
Training vs Validation Recall
ROC Curve Analysis


💾 Model Export

Model saved using Keras:

churn_model.keras
🌐 Streamlit Web Application

An interactive UI built using Streamlit for real-time predictions.

Features:
User-friendly input sliders and dropdowns
Real-time churn probability prediction
Risk classification (High / Low)
Business recommendation engine
Custom UI styling (dark theme)


🧾 Input Features (UI)


Feature	Type
Credit Score	Numeric
Geography	Categorical
Gender	Categorical
Age	Numeric
Tenure	Numeric
Balance	Numeric
NumOfProducts	Numeric
Has Credit Card	Binary
Active Member	Binary


Estimated Salary	Numeric


🔮 Prediction Logic
probability = model.predict(input_scaled)

Decision Rule:

probability > 0.5 → High churn risk
probability ≤ 0.5 → Low churn risk


🧑‍💼 Business Impact

This system helps banks:

Identify at-risk customers early
Reduce customer churn rate
Improve retention strategies
Optimize marketing campaigns
Increase customer lifetime value (CLV)
🛠️ Tech Stack
Machine Learning
TensorFlow / Keras
Scikit-learn
Pandas, NumPy
Visualization
Matplotlib
Deployment
Streamlit


▶️ Run Application
streamlit run app.py


📌 Future Improvements
Replace ANN with XGBoost / Hybrid Model
Add SHAP explainability for predictions
Deploy using Docker + AWS / Azure
Add real-time database integration
Improve feature engineering (behavioral features)


👨‍💻 Author

Bhuwan Adhikari

AI / Robotics Engineer
Interests: Deep Learning, Robotics, Swarm Systems, Autonomous Systems
