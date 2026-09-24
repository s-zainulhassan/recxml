from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    data_dir: Path = Path("data/MINDsmall_train")
    artifacts_dir: Path = Path("artifacts")
    sample_fraction: float = 0.01
    random_state: int = 42
    embedding_size: int = 100
    word2vec_window: int = 5
    word2vec_min_count: int = 1
    word2vec_workers: int = 4
    n_clusters: int = 50
