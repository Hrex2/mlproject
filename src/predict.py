import pandas as pd
import joblib

# load training feature structure
features = joblib.load("models/features.pkl")

def preprocess_input(df):

    # clean column names
    df.columns = df.columns.str.strip()

    # 🔥 IMPORTANT: one-hot encoding
    df = pd.get_dummies(df)

    # 🔥 CRITICAL: match training columns
    df = df.reindex(columns=features, fill_value=0)

    return df


def predict(model, scaler, df):

    df_processed = preprocess_input(df)

    # 🔥 DEBUG (optional)
    print("Final columns:", df_processed.columns.tolist())

    df_scaled = scaler.transform(df_processed)

    preds = model.predict(df_scaled)
    
    print(df_processed.shape)
    return preds

