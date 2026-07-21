import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import load_raw, save_processed

TARGET = "Machine failure"
FAILURE_TYPES = ["TWF", "HDF", "PWF", "OSF", "RNF"]
DROP_COLS = ["UDI", "Product ID"]
NUM_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

COLUMN_RENAME = {
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
}
CAT_FEATURES = ["Type"]
FEATURES = NUM_FEATURES + CAT_FEATURES

RANDOM_STATE = 42


def _encode_type(df: pd.DataFrame) -> pd.DataFrame:
    type_map = {"L": 0, "M": 1, "H": 2}
    df["Type"] = df["Type"].map(type_map)
    return df


def build_features_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = _encode_type(df)
    X = df[FEATURES].copy()
    X = X.rename(columns=COLUMN_RENAME)
    y = df[TARGET].copy()
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def run_preprocessing(save: bool = True) -> dict:
    df = load_raw()
    df = df.drop(columns=DROP_COLS, errors="ignore")
    X, y = build_features_targets(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    if save:
        save_processed(X_train, "X_train")
        save_processed(X_test, "X_test")
        save_processed(y_train.to_frame(name=TARGET), "y_train")
        save_processed(y_test.to_frame(name=TARGET), "y_test")

        full_data = pd.concat([X_train, y_train], axis=1)
        save_processed(full_data, "train_data")
        full_test = pd.concat([X_test, y_test], axis=1)
        save_processed(full_test, "test_data")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


if __name__ == "__main__":
    result = run_preprocessing()
    print(
        f"Train: {result['X_train'].shape}, Test: {result['X_test'].shape}"
    )
    print(
        f"Train failure rate: {result['y_train'].mean():.4f}, "
        f"Test failure rate: {result['y_test'].mean():.4f}"
    )
