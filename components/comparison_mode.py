import streamlit as st
import pandas as pd
from src.evaluate import compare_models
from src.utils import fill_defaults


def run(models, scaler):

    st.subheader("📊 Model Comparison")

    file = st.file_uploader("Upload Dataset", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("### 📂 Preview")
        st.dataframe(df.head())

        if "Attrition" not in df.columns:
            st.error("❌ Dataset must contain 'Attrition' column")
            return

        # 🔥 Fill missing features
        df = fill_defaults(df)

        if st.button("Run Comparison"):

            results = compare_models(models, scaler, df)

            # 📊 Metrics
            st.markdown("## 📊 Metrics")
            st.dataframe(results["metrics"])

            # 🏆 Best Model
            best = results["metrics"].sort_values("Accuracy", ascending=False).iloc[0]
            st.success(
                f"🏆 Best Model: {best['Model']} "
                f"(Accuracy: {best['Accuracy']:.2f})"
            )

            st.markdown("---")

            # 📈 ROC Curve
            st.markdown("## 📈 ROC Curve")
            st.pyplot(results["roc"])

            # 📊 Accuracy Comparison
            st.markdown("## 📊 Accuracy Comparison")
            st.pyplot(results["bar"])

            # 🔥 Confusion Matrix
            st.markdown("## 🔥 Confusion Matrix")
            st.pyplot(results["cm"])