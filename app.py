import streamlit as st
from src.load_model import load_models

from components import single_mode, batch_mode, comparison_mode

models, scaler = load_models()

st.title("📊 Attrition Dashboard")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Single", "Batch", "Comparison"]
)

if mode == "Single":
    single_mode.run(models, scaler)

elif mode == "Batch":
    batch_mode.run(models, scaler)

elif mode == "Comparison":
    comparison_mode.run(models, scaler)   # ✅ FIXED