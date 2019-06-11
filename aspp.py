"""ASPP block."""

import tensorflow as tf
import numpy as np

LAYERS = tf.keras.layers
L2 = tf.keras.regularizers.l2


def sep_conv_bn_relu(inputs,
                     filters=256,
                     kernel_size=3,
                     strides=1,
                     dilation_rate=1,
                     weight_decay=1e-5,
                     batch_normalization=False,
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
            depthwise_regularizer=L2(weight_decay),
            dilation_rate=dilation_rate)(inputs)
        if batch_normalization:
            result = LAYERS.BatchNormalization()(result)
        result = LAYERS.Activation('relu')(result)
        result = LAYERS.Conv2D(
            filters=filters,
            kernel_size=1,
            use_bias=False,
            kernel_regularizer=L2(weight_decay))(result)
        if batch_normalization:
            result = LAYERS.BatchNormalization()(result)
        result = LAYERS.Activation('relu')(result)

    return result


def aspp(inputs, input_shape, weight_decay=1e-5, batch_normalization=False):
    """the DeepLabv3 ASPP module"""
    output_stride = 32

    # Employ a 1x1 convolution.
    b_0 = LAYERS.Conv2D(
        filters=256,
        kernel_size=1,
        padding='same',
        use_bias=False,
        kernel_regularizer=L2(weight_decay),
        name='aspp0')(inputs)
    if batch_normalization:
        b_0 = LAYERS.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b_0)
    b_0 = LAYERS.Activation('relu', name='aspp0_activation')(b_0)

    # Employ 3x3 convolutions with atrous rate = 6
    b_1 = sep_conv_bn_relu(
        inputs=inputs,
        filters=256,
        kernel_size=3,
        strides=1,
        dilation_rate=6,
        weight_decay=weight_decay,
        batch_normalization=batch_normalization,
        name='aspp_block_r6')

    # Employ 3x3 convolutions with atrous rate = 12
    b_2 = sep_conv_bn_relu(
        inputs=inputs,
        filters=256,
        kernel_size=3,
        strides=1,
        dilation_rate=12,
        weight_decay=weight_decay,
        batch_normalization=batch_normalization,
        name='aspp_block_r12')

    # Employ 3x3 convolutions with atrous rate = 18
    b_3 = sep_conv_bn_relu(
        inputs=inputs,
        filters=256,
        kernel_size=3,
        strides=1,
        dilation_rate=18,
        weight_decay=weight_decay,
        batch_normalization=batch_normalization,
        name='aspp_block_r18')

    # Image Feature branch  # TODO: check again!
    pool_height = int(np.ceil(input_shape[0] / output_stride))
    pool_width = int(np.ceil(input_shape[1] / output_stride))
    b_4 = LAYERS.AveragePooling2D(
        pool_size=(pool_height, pool_width), strides=[1, 1],
        padding='same')(inputs)
    b_4 = LAYERS.Conv2D(
        filters=256,
        kernel_size=1,
        padding='same',
        use_bias=False,
        kernel_regularizer=L2(weight_decay),
        name='image_pooling')(inputs)
    if batch_normalization:
        b_4 = LAYERS.BatchNormalization(
            name='image_pooling_BN', epsilon=1e-5)(b_4)
    b_4 = LAYERS.Activation('relu')(b_4)
    # b_4 = LAYERS.UpSampling2D((pool_height, pool_width))(b_4) TODO: Why?

    # concatenate ASPP branches & project
    result = LAYERS.Concatenate()([b_4, b_0, b_1, b_2, b_3])

    result = LAYERS.Conv2D(
        filters=256,
        kernel_size=1,
        name='concat_projection',
        kernel_regularizer=L2(weight_decay))(result)
    if batch_normalization:
        result = LAYERS.BatchNormalization(
            name='final_pooling_BN', epsilon=1e-5)(result)#TODO : additional
    result = LAYERS.Dropout(rate=0.9, name='concat_projection_dropout')(result)

    return result
