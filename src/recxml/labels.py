import numpy as np
import pandas as pd

def create_label_array(news_data: pd.DataFrame, array_size: int, default_label: int,
                       id_column: str = "NewsID_Numeric",
                       label_column: str = "Label") -> np.ndarray:
    labels = np.full(array_size, default_label)
    labels[news_data[id_column]] = news_data[label_column]
    return labels
