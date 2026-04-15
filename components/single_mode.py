import streamlit as st
import pandas as pd
from src.predict import predict, preprocess_input
from src.utils import fill_defaults   # ✅ correct place


def run(models, scaler):

    st.subheader("👤 Employee Prediction (Advanced)")

    # 🔹 Inputs
    age = st.slider("Age", 18, 60)
    gender = st.selectbox("Gender", ["Male", "Female"])
    dept = st.selectbox("Department", ["Sales", "HR", "Research & Development"])

    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist",
        "Laboratory Technician", "Manager",
        "Manufacturing Director"
    ])

    marital = st.selectbox("Marital Status", ["Single", "Married"])
    overtime = st.selectbox("OverTime", ["Yes", "No"])

    income = st.number_input("Monthly Income", 1000, 20000, 5000)
    job_level = st.slider("Job Level", 1, 5)
    experience = st.slider("Total Working Years", 0, 40)

    if st.button("Predict"):

        # ✅ Create DataFrame
        df = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Department": dept,
            "JobRole": job_role,
            "MaritalStatus": marital,
            "OverTime": overtime,
            "MonthlyIncome": income,
            "JobLevel": job_level,
            "TotalWorkingYears": experience
        }])

        # ✅ Fill missing features
        df = fill_defaults(df)

        model = models["Random Forest"]

        # prediction
        pred = predict(model, scaler, df)[0]

        # probability
        df_processed = preprocess_input(df)
        df_scaled = scaler.transform(df_processed)
        prob = model.predict_proba(df_scaled)[0][1]

        # output
        if pred == 1:
            st.error(f"⚠️ High Risk ({prob*100:.1f}%)")
        else:
            st.success(f"✅ Low Risk ({(1-prob)*100:.1f}%)")

        st.metric("Attrition Risk", f"{prob*100:.1f}%")