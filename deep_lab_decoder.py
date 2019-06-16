"""DeepLab v.3+ decoder"""

import tensorflow as tf

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


def decoder(inputs, skip, weight_decay=1e-5, batch_normalization=False):
    """Implementation of DeepLab v.3+ decoder"""

    with tf.name_scope('decoder'):

        skip = LAYERS.Conv2D(filters=48,
                             kernel_size=1,
                             padding='same',
                             use_bias=False,
                             kernel_regularizer=L2(weight_decay),
                             name='feature_projection/Conv2D')(skip)

        # if batch_normalization:
        #     skip = LAYERS.BatchNormalization(
        #         name='feature_projection/bn', epsilon=1e-5)(skip)
        # skip = LAYERS.Activation('relu', name='feature_projection/relu')(skip) # TODO: clean up

        result = LAYERS.UpSampling2D(size=(4, 4),
                                     name='encoder_upsample')(inputs)

        result = LAYERS.Concatenate()([result, skip])

        result = sep_conv_bn_relu(inputs=result,
                                  filters=256,
                                  kernel_size=3,
                                  strides=1,
                                  batch_normalization=batch_normalization,
                                  name='con3x3')
        result = sep_conv_bn_relu(inputs=result,
                                  filters=256,
                                  kernel_size=3,
                                  strides=1,
                                  batch_normalization=batch_normalization,
                                  name='con3x3')

        return result


def get_logits(inputs,
               weight_decay=1e-5,
               batch_normalization=False,
               low_memory=True):
    """Gets the logits for segmentation classes"""

    if low_memory:
        result = inputs
    else:
        result = LAYERS.UpSampling2D(size=(8, 8),
                                     name='logits/final_upsample')(inputs)

    result = LAYERS.Conv2D(
        filters=22,  # TODO: number of classes
        kernel_size=1,
        padding='same',
        kernel_regularizer=L2(weight_decay),
        name='logits/conv2d_classes')(result)
    # if batch_normalization:
    #     result = LAYERS.BatchNormalization(
    #             name='feature_projection/final_BN', epsilon=1e-5)(result)

    if low_memory:
        result = LAYERS.UpSampling2D(size=(8, 8),
                                     name='logits/final_upsample')(result)

    result = LAYERS.Activation('softmax', name='logits/softmax')(result)

    return result
