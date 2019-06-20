""" Deeplabv3+ model"""

import tensorflow as tf
import xception
import aspp
import deep_lab_decoder

MODEL = tf.keras.Model


def deep_lab_v3(inputs, output_stride=16):
    """Implementation of the Deeplabv3+ """
    if output_stride == 32:
        dilation_rates = [6, 12, 18]
    elif output_stride == 16:
        dilation_rates = [12, 24, 36]

    batch_normalization = False
    weight_decay = 0.05

    result, skip = xception.xception_41(
        inputs=inputs,
        weight_decay=weight_decay,
        batch_normalization=batch_normalization,
        output_stride=16)
    result = aspp.aspp(inputs=result,
                       weight_decay=weight_decay,
                       batch_normalization=batch_normalization,
                       dilation_rates=dilation_rates)

    result = deep_lab_decoder.decoder(inputs=result,
                                      skip=skip,
                                      weight_decay=weight_decay,
                                      batch_normalization=batch_normalization)
    result = deep_lab_decoder.get_logits(
        inputs=result,
        weight_decay=weight_decay,
        batch_normalization=batch_normalization,
        low_memory=True)
    model = MODEL(inputs=inputs, outputs=result, name='Deeplabv3plus')
    return model
