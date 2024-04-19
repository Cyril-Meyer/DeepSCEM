# this code is based on the project "TensorFlow Image Segmentation Models" : https://github.com/Cyril-Meyer/tism/
import tensorflow as tf


class UNet:
    def __init__(self,
                 input_shape=(None, None, 1),
                 depth=5,
                 output_classes=2,
                 output_activation='sigmoid',
                 op_dim=2,
                 dropout=0,
                 pool_size=2,
                 name='UNET',
                 transpose=True):
        self.input_shape = input_shape
        self.depth = depth
        self.output_classes = output_classes
        self.output_activation = output_activation
        self.op_dim = op_dim
        self.dropout = dropout
        self.pool_size = pool_size
        self.modelname = name
        self.transpose = transpose

        if op_dim == 2:  # default
            self.conv = tf.keras.layers.Conv2D
            self.conv_t = tf.keras.layers.Conv2DTranspose
            self.ups = tf.keras.layers.UpSampling2D
            self.pool = tf.keras.layers.MaxPool2D
        elif op_dim == 3:
            self.conv = tf.keras.layers.Conv3D
            self.conv_t = tf.keras.layers.Conv3DTranspose
            self.ups = tf.keras.layers.UpSampling3D
            self.pool = tf.keras.layers.MaxPool3D
        else:
            raise ValueError

    def __call__(self, backbone_encoder, backbone_decoder):
        if self.op_dim == 3:
            backbone_encoder.set3D()
            backbone_decoder.set3D()

        # input
        inputs = tf.keras.Input(shape=self.input_shape)
        X = inputs

        # save blocks outputs
        encoder_out = []
        encoder_out_depth = []

        # encoder
        for i in range(self.depth - 1):
            # filters
            X, depth = backbone_encoder(X, i)
            # save output
            encoder_out.append(X)
            encoder_out_depth.append(depth)
            # pooling
            X = self.pool(self.pool_size)(X)

        # center
        # dropout
        if self.dropout > 0:
            X = tf.keras.layers.Dropout(self.dropout)(X)
        # filter
        X, depth = backbone_encoder(X, self.depth - 1)
        # output
        encoder_out.append(X)
        encoder_out_depth.append(depth)

        # decoder
        for i in range(self.depth - 1,  0, -1):
            # up-sample
            if self.pool_size == 2:
                if self.transpose:
                    X = self.conv_t(encoder_out_depth[i - 1], 2, 2, padding='valid')(X)
                else:
                    X = self.ups(2)(X)
            else:
                if self.transpose:
                    X = self.conv_t(encoder_out_depth[i - 1], 2, self.pool_size, padding='same')(X)
                else:
                    X = self.ups(self.pool_size)(X)
            # concatenate
            X = tf.keras.layers.Concatenate()([X, encoder_out[i - 1]])
            # filters
            X, _ = backbone_decoder(X, i-1)

        # output activation
        outputs = self.conv(self.output_classes, 1, activation=self.output_activation, name='output')(X)

        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.modelname)
        return model
