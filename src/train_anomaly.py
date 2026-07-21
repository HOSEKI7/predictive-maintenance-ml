import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest
from src.data_loader import load_processed
from src.preprocessing import TARGET

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RANDOM_STATE = 42


def train_isolation_forest(X_train, y_train, contamination: float = 0.05):
    anomaly_mask = y_train == 0
    X_normal = X_train[anomaly_mask]

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_normal)
    return model


def save_model(model, name: str) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / name
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(name: str):
    return joblib.load(MODELS_DIR / name)


if __name__ == "__main__":
    X_train = load_processed("X_train")
    y_train = load_processed("y_train")[TARGET]

    model = train_isolation_forest(X_train, y_train)
    save_model(model, "isolation_forest.pkl")

    scores = model.decision_function(X_train)
    print(f"Anomaly scores — min: {scores.min():.4f}, max: {scores.max():.4f}")
