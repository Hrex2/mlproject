import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from src.predict import preprocess_input


def compare_models(models, scaler, df):

    X = df.drop("Attrition", axis=1)

    # 🔥 Label fix
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

    # 📈 ROC
    roc_fig, roc_ax = plt.subplots()

    # 📊 Accuracy Bar
    bar_fig, bar_ax = plt.subplots()

    # 🔥 Confusion Matrix
    cm_fig, cm_ax = plt.subplots()

    for name, model in models.items():

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)
        roc_auc = auc(fpr, tpr)
        roc_ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

        # Metrics
        acc = model.score(X_test, y_test)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        })

        # Save last confusion matrix (simple version)
        cm = confusion_matrix(y_test, y_pred)

    # finalize ROC
    roc_ax.legend()
    roc_ax.set_title("ROC Curve")

    # Accuracy Bar Chart
    metrics_df = pd.DataFrame(results)
    bar_ax.bar(metrics_df["Model"], metrics_df["Accuracy"])
    bar_ax.set_title("Accuracy Comparison")

    # Confusion Matrix plot
    cm_ax.imshow(cm)
    cm_ax.set_title("Confusion Matrix")

    return {
        "metrics": metrics_df,
        "roc": roc_fig,
        "bar": bar_fig,
        "cm": cm_fig
    }