import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from src.data_loader import load_processed
from src.preprocessing import TARGET, FAILURE_TYPES
from src.train_classifier import load_model as load_clf
from src.train_anomaly import load_model as load_if

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
RANDOM_STATE = 42


def evaluate_classifier(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    report_dict = classification_report(
        y_test, y_pred, target_names=["Normal", "Failure"], output_dict=True
    )

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Failure"],
                yticklabels=["Normal", "Failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — XGBoost Classifier")

    REPORTS_DIR.mkdir(exist_ok=True)
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Failure"]))

    return {"y_pred": y_pred, "report": report_dict, "confusion_matrix": cm}


def evaluate_anomaly_detector(model, X_test, y_test, anomaly_percentile: float = 5):
    anomaly_scores = model.decision_function(X_test)
    threshold = np.percentile(anomaly_scores, anomaly_percentile)

    y_anomaly_pred = np.where(anomaly_scores <= threshold, 1, 0)

    print(f"\nAnomaly Detection (Isolation Forest):")
    print(f"  Threshold ({anomaly_percentile}th percentile): {threshold:.4f}")
    print(f"  Flagged as anomalies: {y_anomaly_pred.sum()} / {len(y_anomaly_pred)}")

    overlap = (y_anomaly_pred == 1) & (y_test == 1)
    missed = (y_anomaly_pred == 0) & (y_test == 1)
    print(f"  Anomalies that are actual failures: {overlap.sum()}")
    print(f"  Failures missed by anomaly detector: {missed.sum()}")

    return {
        "anomaly_scores": anomaly_scores,
        "y_anomaly_pred": y_anomaly_pred,
        "threshold": threshold,
    }


def compute_risk(y_clf_pred, y_anomaly_pred):
    risk = np.full(len(y_clf_pred), "green", dtype=object)

    red_mask = y_clf_pred == 1
    risk[red_mask] = "red"

    yellow_mask = (y_clf_pred == 0) & (y_anomaly_pred == 1)
    risk[yellow_mask] = "yellow"

    return risk


if __name__ == "__main__":
    X_test = load_processed("X_test")
    y_test = load_processed("y_test")[TARGET]

    clf = load_clf("xgb_classifier.pkl")
    if_model = load_if("isolation_forest.pkl")

    clf_results = evaluate_classifier(clf, X_test, y_test)
    if_results = evaluate_anomaly_detector(if_model, X_test, y_test)

    risk = compute_risk(
        clf_results["y_pred"],
        if_results["y_anomaly_pred"],
    )

    risk_counts = pd.Series(risk).value_counts()
    print(f"\nRisk distribution on test set:\n{risk_counts}")

    test_data = load_processed("test_data")
    test_data["Risk"] = risk

    report_path = REPORTS_DIR / "test_predictions.csv"
    test_data.to_csv(report_path, index=False)
    print(f"Predictions saved to {report_path}")
