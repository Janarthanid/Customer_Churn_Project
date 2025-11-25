import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# ---------------- Load Trained Model ----------------
model = joblib.load("churn_model.pkl")

# Load dataset to extract encoding reference
df = pd.read_csv("data/archive/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.drop("customerID", axis=1, inplace=True)

# Store encoders for each categorical column (IMPORTANT FIX)
encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le  # store separate encoder

# Get feature list
feature_columns = df.drop("Churn", axis=1).columns

# ---------------------- STREAMLIT UI ----------------------
st.title("📊 Customer Churn Prediction App")

st.write("Fill customer details below to predict churn:")

gender = st.selectbox("Gender", encoders["gender"].classes_)
senior = st.selectbox("Senior Citizen (0 = No, 1 = Yes)", [0, 1])
partner = st.selectbox("Partner", encoders["Partner"].classes_)
dependents = st.selectbox("Dependents", encoders["Dependents"].classes_)
tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
total = st.number_input("Total Charges", min_value=0.0, value=100.0)

if st.button("Predict"):

    # Create input row
    input_data = pd.DataFrame(columns=feature_columns)
    
    # Fill with most common values first
    for col in feature_columns:
        input_data.loc[0, col] = df[col].mode()[0]

    # Update user values
    input_data.loc[0, "gender"] = encoders["gender"].transform([gender])[0]
    input_data.loc[0, "SeniorCitizen"] = senior
    input_data.loc[0, "Partner"] = encoders["Partner"].transform([partner])[0]
    input_data.loc[0, "Dependents"] = encoders["Dependents"].transform([dependents])[0]
    input_data.loc[0, "tenure"] = tenure
    input_data.loc[0, "MonthlyCharges"] = monthly
    input_data.loc[0, "TotalCharges"] = total

    # Predict
    prediction = model.predict(input_data)[0]
    proba = model.predict_proba(input_data)[0][1]

    st.subheader("🔍 Prediction Result")

    if prediction == 1:
        st.error(f"🚨 Customer Likely to Churn!")
    else:
        st.success(f"✅ Customer is Not Likely to Churn")

    st.write(f"Confidence Score: **{proba*100:.2f}%**")
