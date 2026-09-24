import pickle
from pathlib import Path
import numpy as np
import pandas as pd

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_behaviors(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "behaviors.tsv"
    return pd.read_table(
        path, header=None,
        names=["ImpressionID", "UserID", "Time", "History", "Impressions"]
    )

def load_news(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "news.tsv"
    return pd.read_table(
        path, header=None,
        names=["NewsID", "Category", "Subcategory", "Title",
               "Abstract", "URL", "Title_Entities", "Abstract_Entites"]
    )

def save_pickle(obj, path: Path):
    ensure_dir(path.parent)
    with path.open("wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)

def save_dataframe(df: pd.DataFrame, path: Path):
    ensure_dir(path.parent)
    df.to_pickle(path)

def save_numpy(array, path: Path):
    ensure_dir(path.parent)
    np.save(path, array)
