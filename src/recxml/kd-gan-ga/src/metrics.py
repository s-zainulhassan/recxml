import numpy as np

def compute_inverse_propensity(Y_train, a=0.55, b=1.5):
    if hasattr(Y_train, "toarray"):
        label_frequency = np.asarray(Y_train.sum(axis=0)).ravel()
    else:
        label_frequency = np.sum(Y_train, axis=0)
    num_samples = Y_train.shape[0]
    C = (np.log(num_samples) - 1.0) * ((b + 1.0) ** a)
    propensity = 1.0 / (1.0 + C * np.power(label_frequency + b, -a))
    return 1.0 / propensity

def precision_at_k(y_true, y_pred, k):
    num_samples = y_true.shape[0]
    score = 0.0
    for i in range(num_samples):
        top_k = np.argsort(y_pred[i])[-k:][::-1]
        score += np.sum(y_true[i][top_k]) / k
    return score / num_samples

def psp_at_k(y_true, y_pred, inverse_propensity, k):
    num_samples = y_true.shape[0]
    score = 0.0
    for i in range(num_samples):
        top_k = np.argsort(y_pred[i])[-k:][::-1]
        numerator = np.sum(y_true[i][top_k] * inverse_propensity[top_k])
        positive_labels = np.where(y_true[i] == 1)[0]
        if len(positive_labels) == 0:
            continue
        ideal = np.sort(inverse_propensity[positive_labels])[::-1][:k]
        denominator = np.sum(ideal)
        if denominator > 0:
            score += numerator / denominator
    return score / num_samples

def dcg_at_k(relevance):
    relevance = np.asarray(relevance)
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return np.sum(relevance / discounts)

def ndcg_at_k(y_true, y_pred, k):
    num_samples = y_true.shape[0]
    score = 0.0
    for i in range(num_samples):
        predicted = np.argsort(y_pred[i])[-k:][::-1]
        dcg = dcg_at_k(y_true[i][predicted])
        ideal = np.sort(y_true[i])[::-1][:k]
        idcg = dcg_at_k(ideal)
        if idcg > 0:
            score += dcg / idcg
    return score / num_samples

def evaluate_predictions(y_true, y_pred, inverse_propensity):
    return {
        "P@1": precision_at_k(y_true, y_pred, 1),
        "P@3": precision_at_k(y_true, y_pred, 3),
        "P@5": precision_at_k(y_true, y_pred, 5),
        "PSP@1": psp_at_k(y_true, y_pred, inverse_propensity, 1),
        "PSP@3": psp_at_k(y_true, y_pred, inverse_propensity, 3),
        "PSP@5": psp_at_k(y_true, y_pred, inverse_propensity, 5),
        "nDCG@1": ndcg_at_k(y_true, y_pred, 1),
        "nDCG@3": ndcg_at_k(y_true, y_pred, 3),
        "nDCG@5": ndcg_at_k(y_true, y_pred, 5),
    }
