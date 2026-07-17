from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch


def evaluate(model, X_test, y_test, device="cpu"):
    
    model.eval()
    with torch.no_grad():
        X_test, y_test = X_test.to(device), y_test.to(device)
        logits = model(X_test)
        preds = torch.argmax(logits, dim=1)

    preds_np = preds.cpu().numpy()
    y_np = y_test.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_np, preds_np),
        "f1": f1_score(y_np, preds_np, zero_division=0),
        "precision": precision_score(y_np, preds_np, zero_division=0),
        "recall": recall_score(y_np, preds_np, zero_division=0),
    }
    return metrics