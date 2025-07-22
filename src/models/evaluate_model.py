from sklearn.metrics import classification_report, accuracy_score
import mlflow

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, output_dict=True)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metrics({
        "precision_avg": report["weighted avg"]["precision"],
        "recall_avg": report["weighted avg"]["recall"],
        "f1_avg": report["weighted avg"]["f1-score"]
    })

    print(f"Accuracy: {acc:.4f}")
    return acc