import pandas as pd
import joblib

# load training feature columns
features = joblib.load("models/features.pkl")

def preprocess_input(data):

    # clean column names
    data.columns = data.columns.str.strip()

    # ensure required columns exist (optional safety)
    if "Salary" not in data.columns:
        data["Salary"] = "Low"

    if "MonthlyHours" not in data.columns:
        data["MonthlyHours"] = 0

    # 🔥 IMPORTANT: use same encoding as training
    data = pd.get_dummies(data)

    # match training columns exactly
    data = data.reindex(columns=features, fill_value=0)

    return data