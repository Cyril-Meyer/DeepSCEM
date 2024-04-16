import tensorflow as tf

import patch


def get_loss(name, multiclass=False):
    name = name.upper()
    if name == 'DICE':
        dice_coef_multi if multiclass else dice_coef
    elif name == 'CROSSENTROPY':
        raise NotImplementedError
    else:
        raise NotImplementedError


def train_model(model, dataset_train, dataset_valid, loss,
                batch_size, patch_size, steps_per_epoch, epochs, validation_steps, callbacks):
    train_img, train_lbl = dataset_train
    valid_img, valmid_lbl = dataset_valid
    # Create patch generators
    gen_train = patch.gen_patch_batch(patch_size, train_img, train_lbl, batch_size=batch_size, augmentation=True)
    gen_valid = patch.gen_patch_batch(patch_size, valid_img, valmid_lbl, batch_size=batch_size, augmentation=True)
    # todo
    # model.fit(...)
    return


def eval_model(model, batch_size, patch_size, evaluation_steps):
    # todo
    raise NotImplementedError


# ----------------------------------------
# Loss functions (code from NeNISt)
# ----------------------------------------
def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    smooth = 0.0001
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def dice_coef_multi(y_true, y_pred, n_label):
    dice = 0
    for index in range(n_label):
        dice += dice_coef(y_true[...,index], y_pred[...,index])
    return dice / n_label
