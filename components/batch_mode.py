import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.predict import predict, preprocess_input
from src.utils import fill_defaults


def run(models, scaler):

    st.subheader("📂 Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        st.write("### 📂 Preview")
        st.dataframe(df.head())

        # 🔥 Fill missing features
        df = fill_defaults(df)

        if st.button("Predict Batch"):

            model = models["Random Forest"]

            # predictions
            preds = predict(model, scaler, df)
            df["Prediction"] = preds

            # probability
            df_processed = preprocess_input(df.drop("Prediction", axis=1))
            df_scaled = scaler.transform(df_processed)
            probs = model.predict_proba(df_scaled)[:, 1]

            df["Attrition_Probability"] = probs

            st.write("### 📊 Results")
            st.dataframe(df)

            # download
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", data=csv, file_name="output.csv")

            st.markdown("---")

            # =======================
            # 📊 GRAPH 1: Distribution
            # =======================
            st.markdown("## 📊 Prediction Distribution")

            fig1, ax1 = plt.subplots()
            counts = df["Prediction"].value_counts()
            ax1.bar(["Stay", "Leave"], counts)
            ax1.set_title("Attrition Count")
            st.pyplot(fig1)

            # =======================
            # 📈 GRAPH 2: Probability Histogram
            # =======================
            st.markdown("## 📈 Attrition Risk Distribution")

            fig2, ax2 = plt.subplots()
            ax2.hist(df["Attrition_Probability"], bins=10)
            ax2.set_title("Probability Distribution")
            st.pyplot(fig2)

            # =======================
            # 📊 GRAPH 3: Department-wise
            # =======================
            if "Department" in df.columns:
                st.markdown("## 📊 Department-wise Attrition")

                dept_data = df.groupby("Department")["Prediction"].mean()

                fig3, ax3 = plt.subplots()
                dept_data.plot(kind="bar", ax=ax3)
                ax3.set_title("Attrition Rate by Department")
                st.pyplot(fig3)

            # =======================
            # 🎯 Summary Metrics
            # =======================
            st.markdown("## 🎯 Summary")

            leave = df["Prediction"].sum()
            total = len(df)

            st.metric("Employees Likely to Leave", f"{leave}/{total}")
            st.metric("Avg Attrition Risk", f"{df['Attrition_Probability'].mean()*100:.1f}%")