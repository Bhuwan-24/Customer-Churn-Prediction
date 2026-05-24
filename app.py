import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="AI Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #0E1117;
}

.title {
    font-size: 42px;
    font-weight: bold;
    color: #4CAFEF;
}

.subtitle {
    font-size: 18px;
    color: #CFCFCF;
}

.metric-box {
    background-color: #1E1E1E;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #333;
}

.prediction-box {
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    font-size: 28px;
    font-weight: bold;
}

.low-risk {
    background-color: rgba(0,255,0,0.15);
    color: #00FF88;
    border: 2px solid #00FF88;
}

.high-risk {
    background-color: rgba(255,0,0,0.15);
    color: #FF4B4B;
    border: 2px solid #FF4B4B;
}
</style>
""", unsafe_allow_html=True)

df = pd.read_csv("Churn_Modelling.csv")

df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

x = df.drop('Exited', axis=1)
y = df['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=40
)

le = LabelEncoder()

x_train['Gender'] = le.fit_transform(x_train['Gender'])
x_test['Gender'] = le.transform(x_test['Gender'])

ohe = OneHotEncoder(
    drop='first',
    handle_unknown='ignore',
    sparse_output=False
)

encoded_train = ohe.fit_transform(x_train[['Geography']])

encoded_train = pd.DataFrame(
    encoded_train,
    columns=ohe.get_feature_names_out(),
    index=x_train.index
)

x_train = x_train.drop('Geography', axis=1)
x_train = pd.concat([x_train, encoded_train], axis=1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)

model = load_model("churn_model.keras")

st.markdown(
    '<p class="title">📊 AI Customer Churn Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict whether a customer is likely to leave the bank using Deep Learning.</p>',
    unsafe_allow_html=True
)

st.divider()


st.sidebar.header("📌 About Project")

st.sidebar.info("""
This AI system predicts whether a customer is likely to churn (leave the bank).

Model Used:
- Artificial Neural Network (ANN)
- TensorFlow / Keras

Accuracy:
- ~85.8%
""")


st.subheader("🧾 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:

    credit_score = st.slider(
        "Credit Score",
        300,
        900,
        650
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.slider(
        "Age",
        18,
        100,
        35
    )

    tenure = st.slider(
        "Tenure (Years)",
        0,
        10,
        5
    )

with col2:

    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=50000.0
    )

    num_products = st.slider(
        "Number of Products",
        1,
        4,
        1
    )

    has_card = st.selectbox(
        "Has Credit Card",
        [0, 1]
    )

    active_member = st.selectbox(
        "Is Active Member",
        [0, 1]
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=100000.0
    )


if st.button("🔍 Predict Churn Risk", use_container_width=True):

 
    gender = le.transform([gender])[0]


    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0


    input_data = pd.DataFrame([[
        credit_score,
        gender,
        age,
        tenure,
        balance,
        num_products,
        has_card,
        active_member,
        estimated_salary,
        geo_germany,
        geo_spain
    ]], columns=x_train.columns)

  
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)

    probability = prediction[0][0]

    st.divider()

    st.subheader("📈 Prediction Result")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Churn Probability</h3>
            <h1>{probability*100:.2f}%</h1>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <h3>Customer Status</h3>
            <h1>{"Likely to Leave" if probability > 0.5 else "Likely to Stay"}</h1>
        </div>
        """, unsafe_allow_html=True)

    st.write("")


    if probability > 0.5:

        st.markdown(f"""
        <div class="prediction-box high-risk">
        ⚠️ HIGH CHURN RISK
        </div>
        """, unsafe_allow_html=True)

        st.warning("""
        Recommended Actions:
        - Offer personalized discounts
        - Improve customer engagement
        - Contact customer support team
        - Provide loyalty benefits
        """)

    else:

        st.markdown(f"""
        <div class="prediction-box low-risk">
        ✅ LOW CHURN RISK
        </div>
        """, unsafe_allow_html=True)

        st.success("""
        Customer appears satisfied and likely to stay with the bank.
        """)

st.divider()

st.caption("Developed using TensorFlow, Keras, Scikit-Learn, and Streamlit")