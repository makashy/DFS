"""ASPP block."""

import tensorflow as tf

layers = tf.keras.layers

def aspp(inputs):
    """the DeepLabv3 ASPP module"""
    OS=16
    
    b0 = layers.Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False, name='aspp0')(inputs)
    b0 = layers.BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
    b0 = layers.Activation('relu', name='aspp0_activation')(b0)
    
    # rate = 6
    b1 = layers.SeparableConv2D(filters=256,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                dilation_rate=6,
                                activation='relu')(inputs)
    # rate = 12
    b2 = layers.SeparableConv2D(filters=256,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                dilation_rate=12,
                                activation='relu')(inputs)
    # rate = 18
    b3 = layers.SeparableConv2D(filters=256,
                                kernel_size=3,
                                strides=1,
                                padding='same',
                                dilation_rate=18,
                                activation='relu')(inputs)
    # Image Feature branch
    # b4 = layers.AveragePooling2D(pool_size=(2, 2))(inputs)# TODO: ?
    b4 = layers.Conv2D(filters=256, kernel_size=1, padding='same',
                       use_bias=False, name='image_pooling')(inputs)
    b4 = layers.BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
    b4 = layers.Activation('relu')(b4)
    # b4 = layers.UpSampling2D((2, 2))(b4)

    # concatenate ASPP branches & project
    x = layers.Concatenate()([b4, b0, b1, b2, b3])

    return x