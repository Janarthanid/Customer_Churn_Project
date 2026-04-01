import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -----------------------------
# Load Model + Encoders + Columns
# -----------------------------
model = joblib.load("churn_model.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Customer Churn App", layout="wide")

# -----------------------------
# Session State
# -----------------------------
if "page" not in st.session_state:
    st.session_state.page = "input"

# -----------------------------
# Custom CSS (🔥 IMPROVED UI)
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    text-align: center;
    color: #6a0dad;
    font-size: 35px;
    font-weight: bold;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# =============================
# PAGE 1 → INPUT FORM
# =============================
if st.session_state.page == "input":

    st.markdown('<div class="title">📝 Customer Details Form</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges", value=50.0)
        total = st.number_input("Total Charges", value=100.0)

    if st.button("➡️ Predict"):

        input_dict = {
            "gender": gender,
            "SeniorCitizen": senior,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "MonthlyCharges": monthly,
            "TotalCharges": total
        }

        input_data = pd.DataFrame([input_dict])

        # Encode categorical
        for col in encoders:
            if col in input_data.columns:
                input_data[col] = encoders[col].transform(input_data[col])

        input_data = input_data.reindex(columns=columns, fill_value=0)

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.session_state.prediction = prediction
        st.session_state.proba = proba

        st.session_state.page = "result"
        st.rerun()

# =============================
# PAGE 2 → DASHBOARD
# =============================
elif st.session_state.page == "result":

    st.markdown('<div class="title">📊 Customer Churn Analysis Dashboard</div>', unsafe_allow_html=True)

    if st.button("⬅️ Back"):
        st.session_state.page = "input"
        st.rerun()

    prediction = st.session_state.prediction
    proba = st.session_state.proba

    # -----------------------------
    # Prediction Result
    # -----------------------------
    st.subheader("🔎 Prediction Result")

    if prediction == 1:
        st.error("🚨 Customer Likely to Churn!")
    else:
        st.success("✅ Customer is Not Likely to Churn")

    st.progress(int(proba * 100))

    # -----------------------------
    # KPI CARDS
    # -----------------------------
    st.subheader("📌 Key Metrics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <h4>Total Customers</h4>
            <h2>7043</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <h4>Churn Probability</h4>
            <h2>{proba*100:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <h4>Avg Monthly Charges</h4>
            <h2>₹64.76</h2>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------
    # CHARTS ROW 1
    # -----------------------------
    st.subheader("📈 Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        ax1.pie([26.5, 73.5], labels=['Churn', 'Not Churn'], autopct='%1.1f%%')
        ax1.set_title("Churn Distribution")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        ax2.scatter([10,20,30,40,50,60,70], [20,40,60,80,90,100,110])
        ax2.set_title("Charges vs Tenure")
        ax2.set_xlabel("Tenure")
        ax2.set_ylabel("Charges")
        st.pyplot(fig2)

    # -----------------------------
    # CHARTS ROW 2
    # -----------------------------
    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots()
        ax3.bar(['Month-to-month', 'One year', 'Two year'], [3000, 1500, 800])
        ax3.set_title("Customers by Contract")
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots()
        ax4.bar(['Fiber', 'DSL', 'No'], [3000, 2000, 1000])
        ax4.set_title("Internet Service Usage")
        st.pyplot(fig4)