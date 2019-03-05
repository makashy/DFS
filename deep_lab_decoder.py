"""DeepLab v.3+ decoder"""

import tensorflow as tf
import numpy as np
layers = tf.keras.layers

def decoder(inputs, skip):
    """Implementation of DeepLab v.3+ decoder"""

    skip = layers.Conv2D(filters=48,
                         kernel_size=1,
                         padding='same',
                         use_bias=False,
                         name='feature_projection0')(skip)

    skip = layers.BatchNormalization(name='feature_projection0_BN', epsilon=1e-5)(skip)

    skip = layers.Activation('relu')(skip)

    result = layers.UpSampling2D(size=(int(np.ceil(4)),
                                              int(np.ceil(4))))(inputs)

    result = layers.Concatenate()([result, skip])

    result = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(result)
    result = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')(result)

    result = layers.Conv2D(filters=21,#TODO : was 21
                           kernel_size=1,
                           padding='same',
                           name='logits_semantic')(result)
    result = layers.UpSampling2D(size=(8, 8))(result)

    return result
