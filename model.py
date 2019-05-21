""" Deeplabv3+ model"""

import tensorflow as tf
import xception
import aspp
import deep_lab_decoder

MODEL = tf.keras.Model


def deep_lab_v3(inputs):
    """Implementation of the Deeplabv3+ """
    batch_normalization = True
    weight_decay = 0.05
    result, skip = xception.xception_71(
        inputs=inputs,
        weight_decay=weight_decay,
        batch_normalization=batch_normalization)
    result = aspp.aspp(
        inputs=result,
        input_shape=[480, 640, 3],
        weight_decay=weight_decay,
        batch_normalization=batch_normalization)
    result = deep_lab_decoder.decoder(
        inputs=result,
        skip=skip,
        weight_decay=weight_decay,
        batch_normalization=batch_normalization)
    model = MODEL(inputs=inputs, outputs=result, name='Deeplabv3+')
    return model
