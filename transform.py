# this code is based on the project "Neural Networks Image Segmentation toolkit" : https://github.com/Cyril-Meyer/NeNISt
import numpy as np
import edt


def normalize_tanh(distance, scale=4.0):
    return np.tanh(distance / scale)


def normalize_sign(distance, scale=2.0):
    return 1 / (1 + np.exp(-(distance / scale)))


def normalize_linear_clip_0_1(distance, scale=8.0):
    return np.clip((distance / scale), 0, 1)


def label_dt(label, anisotropy=None, normalize=normalize_tanh, normalize_scale_pos=5.0, normalize_scale_neg=5.0):
    ndim = len(label.shape)
    # default anisotropy: no anisotropy
    if anisotropy == None:
        anisotropy = (1.0,) * ndim
    # check anisotropy shape
    if not len(anisotropy) == ndim:
        print("ERROR: label_dt invalid anisotropy tuple length")

    distance_pos = edt.edt(label, anisotropy=anisotropy)
    distance_pos = normalize(distance_pos, normalize_scale_pos)

    distance_neg = -edt.edt(1 - label, anisotropy=anisotropy)
    distance_neg = normalize(distance_neg, normalize_scale_neg)

    distance = distance_neg
    distance[label == 1] = distance_pos[label == 1]

    return distance


def label_dt_f16(label, anisotropy=None, normalize=normalize_tanh, normalize_scale_pos=1.0, normalize_scale_neg=1.0):
    ndim = len(label.shape)
    # default anisotropy: no anisotropy
    if anisotropy == None:
        anisotropy = (1.0,) * ndim
    # check anisotropy shape
    if not len(anisotropy) == ndim:
        print("ERROR: label_dt invalid anisotropy tuple length")

    distance_pos = edt.edt(label, anisotropy=anisotropy).astype(np.float16)
    distance_pos = normalize(distance_pos, normalize_scale_pos)

    distance_neg = -edt.edt(1 - label, anisotropy=anisotropy).astype(np.float16)
    distance_neg = normalize(distance_neg, normalize_scale_neg)

    distance = distance_neg
    distance[label == 1] = distance_pos[label == 1]

    return distance


def label_edt(label, anisotropy=None):
    ndim = len(label.shape)
    # default anisotropy: no anisotropy
    if anisotropy == None:
        anisotropy = (1.0,) * ndim
    # check anisotropy shape
    if not len(anisotropy) == ndim:
        print("ERROR: label_dt invalid anisotropy tuple length")

    return edt.edt(label, anisotropy=anisotropy) - edt.edt(1 - label, anisotropy=anisotropy)