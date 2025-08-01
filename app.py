import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction App")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.slider("Age", 0, 80, 25)
sibsp = st.slider("Siblings/Spouses aboard", 0, 8, 0)
parch = st.slider("Parents/Children aboard", 0, 6, 0)
fare = st.number_input("Fare Paid", min_value=0.0, value=30.0)
embarked = st.selectbox("Port of Embarkation", ["C", "Q", "S"])

# Encode inputs
sex_encoded = 1 if sex == "male" else 0
embarked_map = {"C": 0, "Q": 1, "S": 2}
embarked_encoded = embarked_map[embarked]

features = np.array([[pclass, sex_encoded, age, sibsp, parch, fare, embarked_encoded]])

# Predict
if st.button("Predict"):
    result = model.predict(features)[0]
    prob = model.predict_proba(features)[0][result]
    
    if result == 1:
        st.success(f"🎉 Passenger would have **Survived** (Confidence: {prob:.2%})")
    else:
        st.error(f"💀 Passenger would have **Died** (Confidence: {prob:.2%})")
