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

        result = LAYERS.DepthwiseConv2D(kernel_size=kernel_size,
                                        strides=strides,
                                        padding='same',
                                        depth_multiplier=1,
                                        use_bias=False,
                                        depthwise_regularizer=L2(weight_decay),
                                        dilation_rate=dilation_rate)(inputs)
        if batch_normalization:
            result = LAYERS.BatchNormalization()(result)
        result = LAYERS.Activation('relu')(result)
        result = LAYERS.Conv2D(filters=filters,
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
    aspp_1x1 = LAYERS.Conv2D(filters=256,
                             kernel_size=1,
                             padding='same',
                             use_bias=False,
                             kernel_regularizer=L2(weight_decay),
                             name='aspp/1x1/con2d')(inputs)

    if batch_normalization:
        aspp_1x1 = LAYERS.BatchNormalization(name='aspp/1x1/bn',
                                             epsilon=1e-5)(aspp_1x1)

    aspp_1x1 = LAYERS.Activation('relu', name='aspp/1x1/relu')(aspp_1x1)

    # Employ 3x3 convolutions with atrous rate = 6
    aspp_3x3_r6 = sep_conv_bn_relu(inputs=inputs,
                                   filters=256,
                                   kernel_size=3,
                                   strides=1,
                                   dilation_rate=6,
                                   weight_decay=weight_decay,
                                   batch_normalization=batch_normalization,
                                   name='aspp/3x3_r6')

    # Employ 3x3 convolutions with atrous rate = 12
    aspp_3x3_r12 = sep_conv_bn_relu(inputs=inputs,
                                    filters=256,
                                    kernel_size=3,
                                    strides=1,
                                    dilation_rate=12,
                                    weight_decay=weight_decay,
                                    batch_normalization=batch_normalization,
                                    name='aspp/3x3_r12')

    # Employ 3x3 convolutions with atrous rate = 18
    aspp_3x3_r18 = sep_conv_bn_relu(inputs=inputs,
                                    filters=256,
                                    kernel_size=3,
                                    strides=1,
                                    dilation_rate=18,
                                    weight_decay=weight_decay,
                                    batch_normalization=batch_normalization,
                                    name='aspp/3x3_r18')

    # Image Feature branch
    pool_height = int(np.ceil(input_shape[0] / output_stride))
    pool_width = int(np.ceil(input_shape[1] / output_stride))

    aspp_image_features = LAYERS.AveragePooling2D(
        pool_size=[pool_height, pool_width],
        name='aspp/image_features/average_pooling')(inputs)

    aspp_image_features = LAYERS.Conv2D(
        filters=256,
        kernel_size=1,
        padding='same',
        use_bias=False,
        kernel_regularizer=L2(weight_decay),
        name='aspp/image_features/conv2d')(aspp_image_features)

    if batch_normalization:
        aspp_image_features = LAYERS.BatchNormalization(
            name='aspp/image_features/image_pooling_BN',
            epsilon=1e-5)(aspp_image_features)

    aspp_image_features = LAYERS.Activation(
        'relu', name='aspp/image_features/relu')(aspp_image_features)

    aspp_image_features = LAYERS.UpSampling2D(
        (pool_height, pool_width),
        name='aspp/image_features/up_sampling')(aspp_image_features)

    # concatenate ASPP branches & project
    result = LAYERS.Concatenate(name='aspp/concat')([
        aspp_1x1, aspp_3x3_r6, aspp_3x3_r12, aspp_3x3_r18, aspp_image_features
    ])

    result = LAYERS.Conv2D(filters=256,
                           kernel_size=1,
                           name='aspp/conv_1x1',
                           kernel_regularizer=L2(weight_decay))(result)

    result = LAYERS.Dropout(rate=0.9, name='aspp/dropout')(result)

    return result
