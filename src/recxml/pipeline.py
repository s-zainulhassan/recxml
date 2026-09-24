from pathlib import Path
import pandas as pd
import nltk

from .config import Config
from .io import load_behaviors, load_news, save_pickle, save_dataframe, save_numpy, ensure_dir
from .users import create_user_attributes
from .interactions import (
    create_user_news_interactions, add_numeric_ids, clean_interactions,
    build_interaction_matrix,
)
from .mappings import make_reverse_mapping
from .news import prepare_news_data, train_word2vec, cluster_news
from .labels import create_label_array

def run(config: Config):
    ensure_dir(config.artifacts_dir)
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    behaviors = load_behaviors(config.data_dir)
    news = load_news(config.data_dir)

    # User attribute vectors
    news_categories = news[["NewsID", "Category", "Subcategory"]]
    user_attributes = create_user_attributes(behaviors, news_categories)

    # User-news interactions and numeric IDs
    interactions = create_user_news_interactions(
        behaviors, config.sample_fraction, config.random_state
    )
    interactions, user_id_mapping, news_id_mapping = add_numeric_ids(interactions)
    reverse_user_id_mapping = make_reverse_mapping(user_id_mapping)
    reverse_news_id_mapping = make_reverse_mapping(news_id_mapping)

    user_attributes.index.name = "UserID"
    user_attributes = user_attributes.reset_index()
    user_attributes["UserID_Numeric"] = user_attributes["UserID"].map(user_id_mapping)
    user_attributes = user_attributes[
        ["UserID_Numeric"] + [c for c in user_attributes.columns if c != "UserID_Numeric"]
    ]

    # Persist mappings and intermediate data
    save_pickle(user_id_mapping, config.artifacts_dir / "user_id_mapping.pkl")
    save_pickle(news_id_mapping, config.artifacts_dir / "news_id_mapping.pkl")
    save_pickle(reverse_user_id_mapping, config.artifacts_dir / "reverse_user_id_mapping.pkl")
    save_pickle(reverse_news_id_mapping, config.artifacts_dir / "reverse_news_id_mapping.pkl")
    save_dataframe(user_attributes, config.artifacts_dir / "user_attributes.pkl")

    cleaned = clean_interactions(interactions)
    save_dataframe(cleaned, config.artifacts_dir / "user-news-interaction-cleaned.pkl")
    save_dataframe(interactions, config.artifacts_dir / "user-news-interaction.pkl")

    matrix = build_interaction_matrix(cleaned)
    save_numpy(matrix, config.artifacts_dir / "dataset.npy")

    # News embeddings and labels
    news_interacted = interactions["NewsID"].unique()
    news_data = prepare_news_data(news, news_interacted)
    news_data, _ = train_word2vec(
        news_data,
        vector_size=config.embedding_size,
        window=config.word2vec_window,
        min_count=config.word2vec_min_count,
        workers=config.word2vec_workers,
    )
    news_data = cluster_news(news_data, config.n_clusters, config.random_state)
    news_data["NewsID_Numeric"] = news_data["NewsID"].map(news_id_mapping)
    save_dataframe(news_data, config.artifacts_dir / "labelled_news_dataframe.pkl")

    labels = create_label_array(
        news_data,
        array_size=len(news_id_mapping),
        default_label=config.n_clusters,
    )
    save_numpy(labels, config.artifacts_dir / "labels.npy")

    return {
        "behaviors": behaviors,
        "user_attributes": user_attributes,
        "interactions": interactions,
        "interaction_matrix": matrix,
        "news_data": news_data,
        "labels": labels,
    }
