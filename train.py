import tensorflow as tf
import numpy as np

import patch


def train_model(model, dataset_train, dataset_valid, loss,
                batch_size, patch_size, steps_per_epoch, epochs, validation_steps, callbacks, augmentations,
                label_focus=0):
    augmentation_rotation, augmentation_flip = augmentations
    # validation = False if (validation_steps is None or validation_steps <= 0) else True
    train_img, train_lbl = dataset_train
    valid_img, valid_lbl = dataset_valid
    # label focus on train
    train_lbl_id = None
    if label_focus > 0:
        train_lbl_id = []
        for i in range(len(train_lbl)):
            lbl_id = []
            for cla in range(train_lbl[i].shape[-1]):
                lbl_id.append(np.argwhere(train_lbl[i][:, :, :, cla] > 0))
            train_lbl_id.append(lbl_id)
            '''
            lbl_id = patch.create_label_indexes(train_lbl[i], patch_size)
            for j in range(len(lbl_id)):
                np.random.shuffle(lbl_id[j])
            train_lbl_id.append(lbl_id)
            '''
    # Create patch generators
    gen_train = patch.gen_patch_batch(patch_size, train_img, train_lbl, batch_size=batch_size,
                                      augmentation_rotation=augmentation_rotation, augmentation_flip=augmentation_flip,
                                      label_indexes=train_lbl_id, label_indexes_prop=0.75)
    gen_valid = patch.gen_patch_batch(patch_size, valid_img, valid_lbl, batch_size=batch_size,
                                      augmentation_rotation=augmentation_rotation, augmentation_flip=augmentation_flip)

    # Train model
    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss=loss)
    fit_history = model.fit(gen_train,
                            steps_per_epoch=steps_per_epoch,
                            epochs=epochs,
                            validation_data=gen_valid,
                            validation_steps=validation_steps,
                            callbacks=callbacks)
    return model


def eval_model(model, batch_size, patch_size, evaluation_steps):
    # todo
    raise NotImplementedError


# ----------------------------------------
# Loss functions getter
# ----------------------------------------
def get_loss(name, n_classes=1):
    name = name.upper()
    if name == 'DICE':
        return dice_coef_multi(n_classes) if n_classes > 1 else dice_coef
    elif name == 'CROSSENTROPY':
        return tf.keras.losses.CategoricalCrossentropy() if n_classes > 1 else tf.keras.losses.BinaryCrossentropy()
    elif name == 'MEANSQUAREDERROR':
        return tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError


# ----------------------------------------
# Loss functions (code from NeNISt)
# ----------------------------------------
def dice_coef(y_true, y_pred):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(tf.multiply(y_true_f, y_pred_f))
    smooth = 0.0001
    return 1. - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    '''
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), 'float32')
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), 'float32')
    smooth = 128.
    '''


def dice_coef_multi(n_label):
    def loss(y_true, y_pred):
        dice = 0.
        for index in range(n_label):
            dice += dice_coef(y_true[..., index], y_pred[..., index])
        return dice / n_label
    return loss
