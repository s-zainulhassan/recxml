import pandas as pd

def create_user_attributes(behaviors: pd.DataFrame, news_categories: pd.DataFrame) -> pd.DataFrame:
    attributes = behaviors[["UserID", "History"]].copy()
    attributes = attributes.dropna(subset=["History"])
    attributes["History"] = attributes["History"].astype(str)
    attributes["News_interacted"] = attributes["History"].apply(str.split)

    news_mapping = {
        row.NewsID: {"Category": row.Category, "Subcategory": row.Subcategory}
        for row in news_categories.itertuples(index=False)
    }

    user_attributes = {}
    for row in attributes.itertuples(index=False):
        category_count = {}
        subcategory_count = {}

        for news_id in row.News_interacted:
            info = news_mapping.get(news_id.strip())
            if info is None:
                continue
            category = info["Category"]
            subcategory = info["Subcategory"]
            category_count[category] = category_count.get(category, 0) + 1
            subcategory_count[subcategory] = subcategory_count.get(subcategory, 0) + 1

        user_attributes[row.UserID] = {**category_count, **subcategory_count}

    return pd.DataFrame.from_dict(user_attributes, orient="index").fillna(0)
