import joblib

def load_models():
    models = {
        "Logistic Regression": joblib.load("models/model_lr.pkl"),
        "Random Forest": joblib.load("models/model_rf.pkl"),
        "XGBoost": joblib.load("models/model_xgb.pkl")
    }
    scaler = joblib.load("models/scaler.pkl")

    return models, scaler