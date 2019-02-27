"""Xception model."""
from __future__ import absolute_import, division, print_function

import tensorflow as tf

# import numpy as np

layers = tf.keras.layers

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
            residual = layers.SeparableConv2D(filters=depth_list[i],
                                              kernel_size=3,
                                              strides=strides_list[i],
                                              padding='same',#TODO: for i=3 ?
                                              data_format='channels_last',
                                              dilation_rate=dilation_rate_list[i],
                                              depth_multiplier=1
                                              )(residual)

        if skip_connection_type == 'conv':
            shortcut = layers.Conv2D(filters=depth_list[-1],
                                     kernel_size=1,
                                     strides=strides_list[-1],
                                     padding='same',#TODO: ?
                                     dilation_rate=dilation_rate_list[-1],
                                     use_bias=False
                                     )(inputs)

            outputs = residual + shortcut
        elif skip_connection_type == 'sum':
            outputs = residual + inputs
        elif skip_connection_type == 'none':
            outputs = residual
        else:
            raise ValueError('Unsupported skip connection type.')

    return outputs

def xception_71(inputs):
    """Xception-71 model."""

    result = layers.Conv2D(filters=32,
                           kernel_size=3,
                           strides=2,
                           use_bias=False,
                           padding='same')(inputs)
    result = layers.BatchNormalization()(result)
    result = layers.ReLU()(result)

    result = layers.Conv2D(filters=64,
                           kernel_size=3,
                           strides=1,
                           use_bias=False,
                           padding='same')(result)
    result = layers.BatchNormalization()(result)
    result = layers.ReLU()(result)

    result = xception_block(inputs=result,
                            depth_list=[128, 128, 128],
                            strides_list=[1, 1, 2],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            name='entry_flow/block1')

    result = xception_block(inputs=result,# TODO: skip?
                            depth_list=[256, 256, 256],
                            strides_list=[1, 1, 1],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            name='entry_flow/block2')

    result = xception_block(inputs=result,# TODO: skip?
                            depth_list=[256, 256, 256],
                            strides_list=[1, 1, 2],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            name='entry_flow/block3')

    result = xception_block(inputs=result,
                            depth_list=[728, 728, 728],
                            strides_list=[1, 1, 1],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            name='entry_flow/block4')

    result = xception_block(inputs=result,
                            depth_list=[728, 728, 728],
                            strides_list=[1, 1, 2],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            name='entry_flow/block5')

    for i in range(16):
        result = xception_block(inputs=result,
                                depth_list=[728, 728, 728],
                                strides_list=[1, 1, 1],
                                dilation_rate_list=[1, 1, 1],
                                skip_connection_type='sum',
                                name='middle_flow/block'+str(i+1))

    result = xception_block(inputs=result,
                            depth_list=[728, 1024, 1024],
                            strides_list=[1, 1, 2],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            name='exit_flow/block1')

    result = xception_block(inputs=result,
                            depth_list=[1536, 1536, 2048],
                            strides_list=[1, 1, 1],
                            dilation_rate_list=[1, 1, 1],
                            skip_connection_type='conv',
                            name='exit_flow/block2')

    return result
    