import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from src.predict import preprocess_input

def compare_models(models, scaler, df):

    X = df.drop("Attrition", axis=1)

    # 🔥 FIXED LABEL HANDLING
    y = df["Attrition"].map({"Yes": 1, "No": 0})

    y = y.dropna()
    X = X.loc[y.index]

    y = y.astype(int)

    # preprocessing
    X = preprocess_input(X)
    X = scaler.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    fig, ax = plt.subplots()

    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

        results.append({
            "Model": name,
            "Accuracy": model.score(X_test, y_test)
        })

    ax.legend()
    ax.set_title("ROC Curve")

    return {
        "metrics": pd.DataFrame(results),
        "roc": fig
    }