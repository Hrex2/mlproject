import streamlit as st
import pandas as pd
from src.evaluate import compare_models
from src.utils import fill_defaults   # 🔥 NEW

def run(models, scaler):

    st.subheader("📊 Model Comparison")

    file = st.file_uploader("Upload Dataset", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("Preview:", df.head())

        if "Attrition" not in df.columns:
            st.error("❌ Need 'Attrition' column")
            return

        # 🔥 Fill missing features
        df = fill_defaults(df)

        if st.button("Run Comparison"):

            results = compare_models(models, scaler, df)

            st.write("### Metrics")
            st.dataframe(results["metrics"])

            st.write("### ROC Curve")
            st.pyplot(results["roc"])