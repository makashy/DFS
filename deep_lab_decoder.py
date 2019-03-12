"""DeepLab v.3+ decoder"""

import tensorflow as tf

LAYERS = tf.keras.layers
L2 = tf.keras.regularizers.l2


def sep_conv_bn_relu(inputs,
                     filters=256,
                     kernel_size=3,
                     strides=1,
                     dilation_rate=1,
                     name="sepconv_bn"):
    """An separable convolution with batch_normalization and relu
    activation after depthwise and pointwise convolutions"""

    with tf.name_scope(name):

        result = LAYERS.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            depth_multiplier=1,
            use_bias=False,
            depthwise_regularizer=L2(1e-5),
            dilation_rate=dilation_rate)(inputs)
        result = LAYERS.BatchNormalization()(result)
        result = LAYERS.Activation('relu')(result)
        result = LAYERS.Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=L2(1e-5))(result)
        result = LAYERS.BatchNormalization()(result)
        result = LAYERS.Activation('relu')(result)

    return result


def decoder(inputs, skip):
    """Implementation of DeepLab v.3+ decoder"""

    with tf.name_scope('decoder'):

        skip = LAYERS.Conv2D(
            filters=48,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_regularizer=L2(1e-5),
            name='feature_projection/Conv2D')(skip)
        skip = LAYERS.BatchNormalization(
            name='feature_projection/BN', epsilon=1e-5)(skip)
        skip = LAYERS.Activation('relu', name='feature_projection/relu')(skip)

        result = LAYERS.UpSampling2D(size=(4, 4))(inputs)

        result = LAYERS.Concatenate()([result, skip])

        result = sep_conv_bn_relu(
            inputs=result, filters=256, kernel_size=3, strides=1)
        result = sep_conv_bn_relu(
            inputs=result, filters=256, kernel_size=3, strides=1)

        result = LAYERS.Conv2D(
            filters=21, kernel_size=1, padding='same',
            name='logits_semantic')(result)

        result = LAYERS.UpSampling2D(size=(8, 8))(result)

        return result
