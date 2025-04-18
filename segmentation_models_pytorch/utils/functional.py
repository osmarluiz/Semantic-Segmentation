import torch
from sklearn.metrics import jaccard_score
import numpy as np

def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is not None:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device).long()) for x in xs]
    return xs

def _threshold(x, threshold=None):
    return (x > threshold).type(x.dtype) if threshold is not None else x

def iou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    # Ensure tensors are boolean type for bitwise operations
    pr = pr.view(-1).bool()
    gt = gt.view(-1).bool()

    intersection = (gt & pr).sum()
    union = (gt | pr).sum() + eps

    return (intersection + eps) / union

jaccard = iou

def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, ignore_channels=None):
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    pr = pr.view(-1)
    gt = gt.view(-1)
    tp = (gt * pr).sum()
    fp = pr.sum() - tp
    fn = gt.sum() - tp
    score = ((1 + beta ** 2) * tp + eps) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)
    return score

def accuracy(pr, gt, threshold=0.5, ignore_channels=None):
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    tp = torch.sum(gt == pr, dtype=pr.dtype)
    return tp / gt.view(-1).shape[0]

def precision(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    return (tp + eps) / (tp + fp + eps)

def recall(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp
    return (tp + eps) / (tp + fn + eps)