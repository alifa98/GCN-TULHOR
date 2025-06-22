import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_topk_accuracy(logits, label, k):
    """
    Computes Top@K accuracy for a single prediction.

    Args:
        logits (torch.Tensor): Model prediction scores [1, num_classes]
        label (torch.Tensor): True label [1]
        k (int): Top K to evaluate

    Returns:
        int: 1 if correct label is in top K predictions, else 0
    """
    top_k_preds = torch.topk(logits, k=k).indices.squeeze(0)
    return int(label.item() in top_k_preds)


def compute_classification_metrics(y_true, y_pred):
    """
    Computes F1, Precision, Recall (macro).

    Args:
        y_true (List[int]): Ground truth labels
        y_pred (List[int]): Predicted labels

    Returns:
        dict: Macro-averaged F1, Precision, Recall
    """
    return {
        "f1_macro": f1_score(y_true, y_pred, average='macro'),
        "precision_macro": precision_score(y_true, y_pred, average='macro'),
        "recall_macro": recall_score(y_true, y_pred, average='macro')
    }
