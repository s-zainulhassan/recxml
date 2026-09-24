import numpy as np
import pandas as pd

def create_user_news_interactions(behaviors: pd.DataFrame, sample_fraction: float = 0.01,
                                  random_state: int = 42) -> pd.DataFrame:
    data = behaviors.copy()
    selected_rows = int(data.shape[0] * sample_fraction)
    data = data.sample(n=selected_rows, random_state=random_state)
    data = data.drop(["Time", "ImpressionID"], axis=1)

    rows = []
    data["Impressions"] = data["Impressions"].astype(str)
    for row in data.itertuples(index=False):
        for impression in row.Impressions.split():
            rows.append({"UserID": row.UserID, "NewsID": impression})

    interactions = pd.DataFrame(rows)
    interactions["Interaction"] = 1
    interactions.loc[interactions["NewsID"].str.endswith("-0"), "Interaction"] = -1
    interactions["NewsID"] = interactions["NewsID"].str.rstrip("-0").str.rstrip("-1")
    return interactions

def add_numeric_ids(data: pd.DataFrame):
    news_ids = data["NewsID"].unique()
    user_ids = data["UserID"].unique()
    news_id_mapping = {news_id: i for i, news_id in enumerate(news_ids)}
    user_id_mapping = {user_id: i for i, user_id in enumerate(user_ids)}
    out = data.copy()
    out["NewsID_Numeric"] = out["NewsID"].map(news_id_mapping)
    out["UserID_Numeric"] = out["UserID"].map(user_id_mapping)
    return out, user_id_mapping, news_id_mapping

def clean_interactions(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(["UserID", "NewsID"], axis=1)[
        ["NewsID_Numeric", "UserID_Numeric", "Interaction"]
    ]

def build_interaction_matrix(data: pd.DataFrame) -> np.ndarray:
    user_ids = data["UserID_Numeric"].unique()
    news_ids = data["NewsID_Numeric"].unique()
    user_index = {v: i for i, v in enumerate(user_ids)}
    news_index = {v: i for i, v in enumerate(news_ids)}
    matrix = np.zeros((len(news_ids), len(user_ids)))
    for row in data.itertuples(index=False):
        matrix[news_index[row.NewsID_Numeric], user_index[row.UserID_Numeric]] = row.Interaction
    return matrix
