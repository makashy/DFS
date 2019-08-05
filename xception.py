"""Xception model."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

# import numpy as np

LAYERS = tf.keras.layers
L2 = tf.keras.regularizers.l2

BATCH_NORM_DECAY = 0.9997
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_SCALE = True


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


def xception_block(inputs,
                   depth_list,
                   strides_list,
                   dilation_rate_list,
                   skip_connection_type='none',
                   weight_decay=1e-5,
                   batch_normalization=False,
                   name="xception_block"):
    """An Xception module."""

    with tf.name_scope(name):
        residual = inputs

        for i in range(3):
            residual = sep_conv_bn_relu(
                inputs=residual,
                filters=depth_list[i],
                kernel_size=3,
                strides=strides_list[i],
                dilation_rate=dilation_rate_list[i],
                weight_decay=1e-5,
                batch_normalization=batch_normalization)

        if skip_connection_type == 'conv':

            shortcut = LAYERS.Conv2D(
                filters=depth_list[-1],
                kernel_size=1,
                strides=strides_list[-1],
                padding='same',
                dilation_rate=dilation_rate_list[-1],
                use_bias=False,
                kernel_regularizer=L2(weight_decay))(inputs)

            if batch_normalization:
                shortcut = LAYERS.BatchNormalization()(shortcut)
            outputs = LAYERS.Add()([residual, shortcut])

        elif skip_connection_type == 'sum':
            outputs = LAYERS.Add()([residual, inputs])
        elif skip_connection_type == 'none':
            outputs = residual
        else:
            raise ValueError('Unsupported skip connection type.')

    return outputs

def xception_41(inputs, weight_decay=1e-5, batch_normalization=False, output_stride=32):
    """Xception-41 model."""

    result = LAYERS.Conv2D(filters=32,
                           kernel_size=3,
                           strides=2,
                           use_bias=False,
                           kernel_regularizer=L2(weight_decay),
                           padding='same',
                           name='entry_flow/conv32/conv_2d')(inputs)
    if batch_normalization:
        result = LAYERS.BatchNormalization(name='entry_flow/conv32/bn')(result)
    result = LAYERS.ReLU(name='entry_flow/conv32/relu')(result)

    result = LAYERS.Conv2D(filters=64,
                           kernel_size=3,
                           strides=1,
                           use_bias=False,
                           kernel_regularizer=L2(weight_decay),
                           padding='same',
                           name='entry_flow/conv64/conv_2d')(result)
    if batch_normalization:
        result = LAYERS.BatchNormalization(name='entry_flow/conv64/bn')(result)
    result = LAYERS.ReLU(name='entry_flow/conv64/relu')(result)

    result = xception_block(inputs=result,
                            depth_list=[128, 128, 128],
                            strides_list=[1, 1, 2],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            weight_decay=weight_decay,
                            batch_normalization=batch_normalization,
                            name='entry_flow/block1')
    if output_stride == 16:
        skip = result

    result = xception_block(inputs=result,
                            depth_list=[256, 256, 256],
                            strides_list=[1, 1, 2],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            weight_decay=weight_decay,
                            batch_normalization=batch_normalization,
                            name='entry_flow/block2')
    if output_stride == 32:
        skip = result

    result = xception_block(inputs=result,
                            depth_list=[728, 728, 728],
                            strides_list=[1, 1, 2],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            weight_decay=weight_decay,
                            batch_normalization=batch_normalization,
                            name='entry_flow/block3')

    for i in range(8):
        result = xception_block(inputs=result,
                                depth_list=[728, 728, 728],
                                strides_list=[1, 1, 1],
                                dilation_rate_list=[1, 1, 1],
                                skip_connection_type='sum',
                                weight_decay=weight_decay,
                                batch_normalization=batch_normalization,
                                name='middle_flow/block' + str(i + 1))

    if output_stride == 16:
        strides_list = [1, 1, 1]
        dilation_rate_list = [2, 2, 2]
    elif output_stride == 32:
        strides_list = [1, 1, 2]
        dilation_rate_list = [1, 1, 1]
    result = xception_block(inputs=result,
                            depth_list=[728, 1024, 1024],
                            strides_list=strides_list,
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            weight_decay=weight_decay,
                            batch_normalization=batch_normalization,
                            name='exit_flow/block1')

    result = xception_block(inputs=result,
                            depth_list=[1536, 1536, 2048],
                            strides_list=[1, 1, 1],
                            dilation_rate_list=dilation_rate_list,
                            skip_connection_type='none',
                            weight_decay=weight_decay,
                            batch_normalization=batch_normalization,
                            name='exit_flow/block2')

    return result, skip


def xception_71(inputs, weight_decay=1e-5, batch_normalization=False):
    """Xception-71 model."""

    result = LAYERS.Conv2D(filters=32,
                           kernel_size=3,
                           strides=2,
                           use_bias=False,
                           kernel_regularizer=L2(weight_decay),
                           padding='same',
                           name='entry_flow/conv32/conv_2d')(inputs)
    if batch_normalization:
        result = LAYERS.BatchNormalization(name='entry_flow/conv32/bn')(result)
    result = LAYERS.ReLU(name='entry_flow/conv32/relu')(result)

    result = LAYERS.Conv2D(filters=64,
                           kernel_size=3,
                           strides=1,
                           use_bias=False,
                           kernel_regularizer=L2(weight_decay),
                           padding='same',
                           name='entry_flow/conv64/conv_2d')(result)
    if batch_normalization:
        result = LAYERS.BatchNormalization(name='entry_flow/conv64/bn')(result)
    result = LAYERS.ReLU(name='entry_flow/conv64/relu')(result)

    result = xception_block(inputs=result,
                            depth_list=[128, 128, 128],
                            strides_list=[1, 1, 2],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            weight_decay=weight_decay,
                            batch_normalization=batch_normalization,
                            name='entry_flow/block1')

    result = xception_block(inputs=result,
                            depth_list=[256, 256, 256],
                            strides_list=[1, 1, 1],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            weight_decay=weight_decay,
                            batch_normalization=batch_normalization,
                            name='entry_flow/block2')

    result = xception_block(inputs=result,
                            depth_list=[256, 256, 256],
                            strides_list=[1, 1, 2],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            weight_decay=weight_decay,
                            batch_normalization=batch_normalization,
                            name='entry_flow/block3')
    skip = result

    result = xception_block(inputs=result,
                            depth_list=[728, 728, 728],
                            strides_list=[1, 1, 1],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            weight_decay=weight_decay,
                            batch_normalization=batch_normalization,
                            name='entry_flow/block4')

    result = xception_block(inputs=result,
                            depth_list=[728, 728, 728],
                            strides_list=[1, 1, 2],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            weight_decay=weight_decay,
                            batch_normalization=batch_normalization,
                            name='entry_flow/block5')

    for i in range(16):
        result = xception_block(inputs=result,
                                depth_list=[728, 728, 728],
                                strides_list=[1, 1, 1],
                                dilation_rate_list=[1, 1, 1],
                                skip_connection_type='sum',
                                weight_decay=weight_decay,
                                batch_normalization=batch_normalization,
                                name='middle_flow/block' + str(i + 1))

    result = xception_block(inputs=result,
                            depth_list=[728, 1024, 1024],
                            strides_list=[1, 1, 2],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            weight_decay=weight_decay,
                            batch_normalization=batch_normalization,
                            name='exit_flow/block1')

    result = xception_block(inputs=result,
                            depth_list=[1536, 1536, 2048],
                            strides_list=[1, 1, 1],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='none',
                            weight_decay=weight_decay,
                            batch_normalization=batch_normalization,
                            name='exit_flow/block2')

    return result, skip
