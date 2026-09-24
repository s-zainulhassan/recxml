import re
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def prepare_news_data(news_data: pd.DataFrame, interacted_news_ids) -> pd.DataFrame:
    news_data = news_data.drop(
        ["URL", "Title_Entities", "Abstract_Entites"], axis=1
    )
    news_data = news_data[news_data["NewsID"].isin(interacted_news_ids)].copy()
    news_data.dropna(subset=["Title", "Abstract"], inplace=True)
    news_data["Processed_Text"] = (
        news_data["Category"].apply(preprocess_text) + " " +
        news_data["Subcategory"].apply(preprocess_text) + " " +
        news_data["Title"].apply(preprocess_text) + " " +
        news_data["Abstract"].apply(preprocess_text)
    )
    return news_data

def train_word2vec(news_data: pd.DataFrame, vector_size=100, window=5,
                   min_count=1, workers=4):
    model = Word2Vec(
        sentences=news_data["Processed_Text"].str.split(),
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
    )

    def get_embedding(words):
        vectors = [model.wv[word] for word in words if word in model.wv]
        return sum(vectors) / len(vectors) if vectors else None

    news_data = news_data.copy()
    news_data["Embeddings"] = news_data["Processed_Text"].str.split().apply(get_embedding)
    return news_data.dropna(subset=["Embeddings"]), model

def cluster_news(news_data: pd.DataFrame, n_clusters=50, random_state=42) -> pd.DataFrame:
    news_data = news_data.copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(list(news_data["Embeddings"]))
    news_data["Label"] = kmeans.labels_
    return news_data
