import streamlit as st
import pandas as pd
from src.predict import predict
from src.utils import fill_defaults   # 🔥 NEW

def run(models, scaler):

    st.subheader("📂 Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("### Preview")
        st.dataframe(df.head())

        # 🔥 Fill missing features automatically
        df = fill_defaults(df)

        if st.button("Predict Batch"):

            preds = predict(models["Random Forest"], scaler, df)

            df["Prediction"] = preds

            st.write("### Results")
            st.dataframe(df)

            # download
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                "Download Results",
                data=csv,
                file_name="output.csv",
                mime="text/csv"
            )

            # summary
            leave = df["Prediction"].sum()
            total = len(df)

            st.metric("Attrition Risk Count", f"{leave}/{total}")