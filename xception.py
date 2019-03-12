"""Xception model."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

# import numpy as np

LAYERS = tf.keras.layers
L2 = tf.keras.regularizers.l2

BATCH_NORM_DECAY = 0.9997
BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_SCALE = True


def xception_block(inputs,
                   depth_list,
                   strides_list,
                   dilation_rate_list,
                   skip_connection_type='none',
                   name="xception_block"):
    """An Xception module."""

    with tf.name_scope(name):
        residual = inputs

        for i in range(3):
            residual = LAYERS.Activation('relu')(residual)

            residual = LAYERS.DepthwiseConv2D(
                kernel_size=3,
                strides=strides_list[i],
                padding='same',
                depth_multiplier=1,
                use_bias=False,
                depthwise_regularizer=L2(1e-5),
                dilation_rate=dilation_rate_list[i])(residual)

            residual = LAYERS.BatchNormalization()(residual)

            residual = LAYERS.Conv2D(
                filters=depth_list[i],
                kernel_size=1,
                use_bias=False,
                kernel_regularizer=L2(1e-5))(residual)

            residual = LAYERS.BatchNormalization()(residual)

        if skip_connection_type == 'conv':

            shortcut = LAYERS.Conv2D(
                filters=depth_list[-1],
                kernel_size=1,
                strides=strides_list[-1],
                padding='same',
                dilation_rate=dilation_rate_list[-1],
                use_bias=False,# TODO: False?
                kernel_regularizer=L2(1e-5))(inputs)

            shortcut = LAYERS.BatchNormalization()(shortcut)
            outputs = LAYERS.Add()([residual, shortcut])

        elif skip_connection_type == 'sum':
            outputs = LAYERS.Add()([residual, inputs])
        elif skip_connection_type == 'none':
            outputs = residual
        else:
            raise ValueError('Unsupported skip connection type.')

    return outputs


def xception_71(inputs):
    """Xception-71 model."""

    result = LAYERS.Conv2D(
        filters=32, kernel_size=3, strides=2, use_bias=False,
        padding='same')(inputs)
    result = LAYERS.BatchNormalization()(result)
    result = LAYERS.ReLU()(result)

    result = LAYERS.Conv2D(
        filters=64, kernel_size=3, strides=1, use_bias=False,
        padding='same')(result)
    result = LAYERS.BatchNormalization()(result)
    result = LAYERS.ReLU()(result)

    result = xception_block(
        inputs=result,
        depth_list=[128, 128, 128],
        strides_list=[1, 1, 2],
        dilation_rate_list=[1, 1, 1],
        skip_connection_type='conv',
        name='entry_flow/block1')

    result = xception_block(
        inputs=result,
        depth_list=[256, 256, 256],
        strides_list=[1, 1, 1],
        dilation_rate_list=[1, 1, 1],
        skip_connection_type='conv',
        name='entry_flow/block2')

    result = xception_block(
        inputs=result,
        depth_list=[256, 256, 256],
        strides_list=[1, 1, 2],
        dilation_rate_list=[1, 1, 1],
        skip_connection_type='conv',
        name='entry_flow/block3')
    skip = result

    result = xception_block(
        inputs=result,
        depth_list=[728, 728, 728],
        strides_list=[1, 1, 1],
        dilation_rate_list=[1, 1, 1],
        skip_connection_type='conv',
        name='entry_flow/block4')

    result = xception_block(
        inputs=result,
        depth_list=[728, 728, 728],
        strides_list=[1, 1, 2],
        dilation_rate_list=[1, 1, 1],
        skip_connection_type='conv',
        name='entry_flow/block5')

    for i in range(16):
        result = xception_block(
            inputs=result,
            depth_list=[728, 728, 728],
            strides_list=[1, 1, 1],
            dilation_rate_list=[1, 1, 1],
            skip_connection_type='sum',
            name='middle_flow/block' + str(i + 1))

    result = xception_block(
        inputs=result,
        depth_list=[728, 1024, 1024],
        strides_list=[1, 1, 2],
        dilation_rate_list=[1, 1, 1],
        skip_connection_type='conv',
        name='exit_flow/block1')

    result = xception_block(# TODO: additional ReLU?
        inputs=result,
        depth_list=[1536, 1536, 2048],
        strides_list=[1, 1, 1],
        dilation_rate_list=[1, 1, 1],
        skip_connection_type='conv',
        name='exit_flow/block2')

    return result, skip
