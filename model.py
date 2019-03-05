""" Deeplabv3+ model"""

import tensorflow as tf
import xception
import aspp
import deep_lab_decoder

MODEL = tf.keras.Model

def deep_lab_v3(inputs):
    """Implementation of the Deeplabv3+ """

    result, skip = xception.xception_71(inputs=inputs)
    result = aspp.aspp(inputs=result)
    result = deep_lab_decoder.decoder(inputs=result, skip=skip)
    model = MODEL(inputs=inputs, outputs=result, name='Deeplabv3+')
    return model
