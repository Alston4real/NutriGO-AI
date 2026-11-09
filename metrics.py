import numpy as np


def iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else np.logical_and(pred, gt).sum() / union


def miou(pred_masks, gt_masks, labels):
    """Mean over classes of the mean per-image IoU for that class."""
    per = {}
    for p, g, l in zip(pred_masks, gt_masks, labels):
        per.setdefault(l, []).append(iou(p, g))
    per_class = {l: float(np.mean(v)) for l, v in per.items()}
    return float(np.mean(list(per_class.values()))), per_class


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(true, float))))


def mape(pred, true):
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    return float(np.mean(np.abs(pred - true) / np.maximum(true, 1e-9)) * 100)


def topk_accuracy(score_rows, true_idx, k=1):
    hits = [t in np.argsort(s)[::-1][:k] for s, t in zip(score_rows, true_idx)]
    return float(np.mean(hits))


def false_positive_rate(n_false_positive, n_predictions):
    return 0.0 if n_predictions == 0 else n_false_positive / n_predictions


def coverage(low, high, true):
    """Share of ground-truth values that fall inside the predicted range."""
    low, high, true = (np.asarray(a, float) for a in (low, high, true))
    return float(np.mean((true >= low) & (true <= high)))
