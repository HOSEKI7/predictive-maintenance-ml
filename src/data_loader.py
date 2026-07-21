import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_PATH = DATA_DIR / "ai4i2020.csv"
PROCESSED_DIR = DATA_DIR / "processed"


def load_raw(path: str | Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def save_processed(df: pd.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(exist_ok=True)
    df.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)


def load_processed(name: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
