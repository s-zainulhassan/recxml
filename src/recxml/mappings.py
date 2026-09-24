def map_user_id(user_id, user_id_mapping):
    return user_id_mapping.get(user_id)

def map_news_id(news_id, news_id_mapping):
    return news_id_mapping.get(news_id)

def reverse_map_user_id(user_id_numeric, reverse_user_id_mapping):
    return reverse_user_id_mapping.get(user_id_numeric)

def reverse_map_news_id(news_id_numeric, reverse_news_id_mapping):
    return reverse_news_id_mapping.get(news_id_numeric)

def make_reverse_mapping(mapping):
    return {numeric_id: original_id for original_id, numeric_id in mapping.items()}
