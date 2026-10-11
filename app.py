import pickle
import pandas as pd
import streamlit as st

with open("model.pkl", "rb") as f:
    saved = pickle.load(f)

models = saved["models"]
scaler = saved["scaler"]
encoders = saved["encoders"]
columns = saved["columns"]
defaults = saved["defaults"]
diseases = saved["diseases"]

st.title("Chronic Disease Prediction System")
st.write("Enter the patient details below.")

# User input
inputs = {}
for col in columns:
    if col in encoders:
        inputs[col] = st.selectbox(col, list(encoders[col].classes_))
    else:
        inputs[col] = st.number_input(col, value=float(defaults[col]))

# Prediction
if st.button("Predict"):
    row = pd.DataFrame([inputs])
    for col, enc in encoders.items():
        row[col] = enc.transform(row[col])
    row = scaler.transform(row[columns])

    # Results: risk rate for each disease
    results = []
    for d in diseases:
        risk = models[d].predict_proba(row)[0][1] * 100
        results.append({"Disease": d.replace("Had", ""), "Risk (%)": round(risk, 1)})

    st.subheader("Predicted risk rate")
    st.table(pd.DataFrame(results))