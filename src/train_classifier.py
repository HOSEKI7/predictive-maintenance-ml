import joblib
import xgboost as xgb
from pathlib import Path
from src.data_loader import load_processed
from src.preprocessing import TARGET

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
RANDOM_STATE = 42


def train_xgb(X_train, y_train, scale_pos_weight: float | None = None):
    if scale_pos_weight is None:
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        scale_pos_weight = neg / pos

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    model.fit(X_train, y_train)
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

    model = train_xgb(X_train, y_train)
    save_model(model, "xgb_classifier.pkl")

    y_pred = model.predict(X_train)
    train_acc = (y_pred == y_train).mean()
    print(f"Train accuracy: {train_acc:.4f}")
