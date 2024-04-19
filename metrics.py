import numpy as np


def iou(true, pred):
    true = true.astype(np.bool)
    pred = pred.astype(np.bool)
    I = np.sum(true * pred)
    U = np.sum(true + pred)
    return I/U


def f1(true, pred):
    true = true.astype(np.bool)
    pred = pred.astype(np.bool)
    I = np.sum(true * pred)
    U = np.sum(true + pred)
    return (2*I)/(U+I)


jaccard = iou
dice = f1
