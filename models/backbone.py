# this code is based on the project "TensorFlow Image Segmentation Models" : https://github.com/Cyril-Meyer/tism/
import tensorflow as tf


class VGG:
    def __init__(self,
                 initial_block_depth=32,
                 initial_block_length=2,
                 activation='relu',
                 kernel_size=3,
                 normalization='BatchNormalization'):
        self.initial_block_depth = initial_block_depth
        self.initial_block_length = initial_block_length
        self.activation = activation
        self.kernel_size = kernel_size
        self.normalization = normalization

        self.conv = tf.keras.layers.Conv2D

    def set3D(self):
        self.conv = tf.keras.layers.Conv3D

    def __call__(self, X, level):
        block_depth = self.initial_block_depth * 2 ** level

        for _ in range(self.initial_block_length):
            X = self.conv(filters=block_depth, kernel_size=self.kernel_size,
                          kernel_initializer='he_normal', padding='same')(X)
            X = tf.keras.layers.Activation(self.activation)(X)

            if self.normalization == 'BatchNormalization':
                X = tf.keras.layers.BatchNormalization()(X)
            elif self.normalization == 'LayerNormalization':
                X = tf.keras.layers.LayerNormalization()(X)

        return X, block_depth


class ResBlock:
    def __init__(self, backbone=VGG()):
        self.backbone = backbone
        self.conv = tf.keras.layers.Conv2D

    def set3D(self):
        self.conv = tf.keras.layers.Conv3D
        self.backbone.set3D()

    def __call__(self, X, level):
        out, block_depth = self.backbone(X, level)

        res = self.conv(filters=block_depth, kernel_size=1)(X)
        res = tf.keras.layers.Activation('relu')(res)
        res = tf.keras.layers.BatchNormalization()(res)

        X = tf.keras.layers.Add()([res, out])
        X = tf.keras.layers.Activation('relu')(X)

        return X, block_depth
