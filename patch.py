# this code is based on the project "Neural Networks Image Segmentation toolkit" : https://github.com/Cyril-Meyer/NeNISt
from random import randint, random
import numpy as np


def check_valid(image, label):
    if type(image) is list and type(label) is list:
        if not len(image) == len(label):
            return False
        for i in range(len(image)):
            if not image[i].shape[:-1] == label[i].shape[:-1] and len(image[i].shape) == 4:
                return False
    elif type(image) is np.ndarray and type(label) is np.ndarray:
        return image.shape[:-1] == label.shape[:-1] and len(image.shape) == 4
    else:
        return False
    return True


def create_label_indexes(label, patch_size):
    label_indexes = []
    if len(patch_size) == 2:
        pz = 0
        py, px = patch_size
    elif len(patch_size) == 3:
        pz, py, px = patch_size
    else:
        raise ValueError

    for i in range(label.shape[-1]):
        mz, my, mx, _ = label.shape
        label_indexes.append(np.argwhere(label[:mz - pz, :my - py, :mx - px, i] > 0))
    return label_indexes


def gen_patch_2d_batch(patch_size, image, label, batch_size, augmentation, label_indexes, label_indexes_prop):
    n_channel = image[0].shape[-1]
    n_label = label[0].shape[-1]
    image_dtype = image[0].dtype
    label_dtype = label[0].dtype
    patch_size_y, patch_size_x = patch_size

    batch_image = np.zeros((batch_size,) + tuple(patch_size) + (n_channel,), dtype=image_dtype)
    batch_label = np.zeros((batch_size,) + tuple(patch_size) + (n_label,), dtype=label_dtype)
    img = image
    lbl = label
    if label_indexes is not None:
        lbli = label_indexes
    while True:
        batch_image.fill(0)
        for i in range(batch_size):
            if type(image) is list:
                n = randint(0, len(image) - 1)
                img = image[n]
                lbl = label[n]
                if label_indexes is not None:
                    lbli = label_indexes[n]
            if label_indexes is not None and label_indexes_prop > random():
                cla = randint(0, len(lbli) - 1)
                r = randint(0, len(lbli[cla]) - 1)
                z, y, x = lbli[cla][r]
                y = np.clip(y - (patch_size_y // 2), 0, img.shape[1] - patch_size_y)
                x = np.clip(x - (patch_size_x // 2), 0, img.shape[2] - patch_size_x)
                z, y, x = int(z), int(y), int(x)
            else:
                x = randint(0, img.shape[2] - patch_size_x)
                y = randint(0, img.shape[1] - patch_size_y)
                z = randint(0, img.shape[0] - 1)

            batch_image[i, :, :, :] = img[z, y:y + patch_size_y, x:x + patch_size_x, :]
            batch_label[i, :, :, :] = lbl[z, y:y + patch_size_y, x:x + patch_size_x, :]

            if augmentation:
                if patch_size_y == patch_size_x:
                    rot = randint(0, 3)
                    batch_image[i, :, :] = np.rot90(batch_image[i, :, :], rot)
                    batch_label[i, :, :] = np.rot90(batch_label[i, :, :], rot)

                if randint(0, 1) == 1:
                    batch_image[i, :, :] = np.fliplr(batch_image[i, :, :])
                    batch_label[i, :, :] = np.fliplr(batch_label[i, :, :])

                if randint(0, 1) == 1:
                    batch_image[i, :, :] = np.flipud(batch_image[i, :, :])
                    batch_label[i, :, :] = np.flipud(batch_label[i, :, :])

        yield batch_image, batch_label


def gen_patch_3d_batch(patch_size, image, label, batch_size, augmentation, label_indexes, label_indexes_prop):
    n_channel = image[0].shape[-1]
    n_label = label[0].shape[-1]
    image_dtype = image[0].dtype
    label_dtype = label[0].dtype
    patch_size_z, patch_size_y, patch_size_x = patch_size

    batch_image = np.zeros((batch_size,) + tuple(patch_size) + (n_channel,), dtype=image_dtype)
    batch_label = np.zeros((batch_size,) + tuple(patch_size) + (n_label,), dtype=label_dtype)
    img = image
    lbl = label
    if label_indexes is not None:
        lbli = label_indexes
    while True:
        batch_image.fill(0)
        for i in range(batch_size):
            if type(image) is list:
                n = randint(0, len(image) - 1)
                img = image[n]
                lbl = label[n]
                if label_indexes is not None:
                    lbli = label_indexes[n]
            if label_indexes is not None and label_indexes_prop > random():
                cla = randint(0, len(lbli) - 1)
                r = randint(0, len(lbli[cla]) - 1)
                z, y, x = lbli[cla][r]
                z = np.clip(z - (patch_size_z // 2), 0, img.shape[0] - patch_size_z)
                y = np.clip(y - (patch_size_y // 2), 0, img.shape[1] - patch_size_y)
                x = np.clip(x - (patch_size_x // 2), 0, img.shape[2] - patch_size_x)
                z, y, x = int(z), int(y), int(x)
            else:
                x = randint(0, img.shape[2] - patch_size_x)
                y = randint(0, img.shape[1] - patch_size_y)
                z = randint(0, img.shape[0] - patch_size_z)

            batch_image[i, :, :, :, :] = img[z:z + patch_size_z, y:y + patch_size_y, x:x + patch_size_x, :]
            batch_label[i, :, :, :, :] = lbl[z:z + patch_size_z, y:y + patch_size_y, x:x + patch_size_x, :]

            if augmentation:
                if patch_size_z == patch_size_x:
                    rot = randint(0, 3)
                    batch_image[i, :, :] = np.rot90(batch_image[i, :, :, :], rot, axes=(0, 2))
                    batch_label[i, :, :] = np.rot90(batch_label[i, :, :, :], rot, axes=(0, 2))
                if patch_size_z == patch_size_y:
                    rot = randint(0, 3)
                    batch_image[i, :, :] = np.rot90(batch_image[i, :, :, :], rot, axes=(0, 1))
                    batch_label[i, :, :] = np.rot90(batch_label[i, :, :, :], rot, axes=(0, 1))
                if patch_size_y == patch_size_x:
                    rot = randint(0, 3)
                    batch_image[i, :, :] = np.rot90(batch_image[i, :, :, :], rot, axes=(1, 2))
                    batch_label[i, :, :] = np.rot90(batch_label[i, :, :, :], rot, axes=(1, 2))

                if randint(0, 1) == 1:
                    batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 0)
                    batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 0)

                if randint(0, 1) == 1:
                    batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 1)
                    batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 1)

                if randint(0, 1) == 1:
                    batch_image[i, :, :, :] = np.flip(batch_image[i, :, :, :], 2)
                    batch_label[i, :, :, :] = np.flip(batch_label[i, :, :, :], 2)

        yield batch_image, batch_label


def gen_patch_batch(patch_size, image, label, batch_size=32, augmentation=True, label_indexes=None,
                    label_indexes_prop=1.0):
    gen = None
    if not (len(patch_size) == 2 or len(patch_size) == 3):
        raise ValueError
    if not check_valid(image, label):
        raise ValueError
    if len(patch_size) == 2:
        gen = gen_patch_2d_batch(patch_size, image, label, batch_size, augmentation, label_indexes, label_indexes_prop)
    elif len(patch_size) == 3:
        gen = gen_patch_3d_batch(patch_size, image, label, batch_size, augmentation, label_indexes, label_indexes_prop)
    else:
        raise ValueError
    return gen


def crop_gen_patch_2d_batch(gen, crop_in, crop_out):
    while True:
        X, Y = next(gen)
        y_min = crop_in[0][0]
        y_max = X.shape[1] - crop_in[0][1]
        x_min = crop_in[1][0]
        x_max = X.shape[2] - crop_in[1][1]
        X = X[:, y_min:y_max, x_min:x_max]

        y_min = crop_out[0][0]
        y_max = Y.shape[1] - crop_out[0][1]
        x_min = crop_out[1][0]
        x_max = Y.shape[2] - crop_out[1][1]
        Y = Y[:, y_min:y_max, x_min:x_max]

        yield X, Y


def gen_to_multiple_outputs(gen):
    while True:
        X, Y = next(gen)

        Y_ = []
        for c in range(Y.shape[-1]):
            Y_.append(Y[..., c:c + 1])

        yield X, Y_
