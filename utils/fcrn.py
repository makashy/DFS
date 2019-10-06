"""FCRN model:
[1] I. Laina, C. Rupprecht, V. Belagiannis, F. Tombari, and N. Navab,
“Deeper Depth Prediction with Fully Convolutional Residual Networks,”
arXiv:1606.00373 [cs], Jun. 2016.
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras.layers import (  # pylint: disable=import-error
    Activation, BatchNormalization, Conv2D, Input, MaxPool2D)

ACTIVATION = 'selu'


def baseline(input_tensor, input_filter, output_filter, block_name):
    "Baseline block"
    with tf.name_scope(block_name):
        result = Conv2D(filters=input_filter, kernel_size=1,
                        padding='same')(input_tensor)
        result = BatchNormalization()(result)
        result = Activation(ACTIVATION)(result)

        result = Conv2D(filters=input_filter, kernel_size=3,
                        padding='same')(input_tensor)
        result = BatchNormalization()(result)
        result = Activation(ACTIVATION)(result)

        result = Conv2D(filters=output_filter, kernel_size=1,
                        padding='same')(input_tensor)
        result = BatchNormalization()(result)
        result = tf.keras.layers.add([result, input_tensor])
        result = Activation(ACTIVATION)(result)

    return result


def bottleneck(input_tensor, input_filter, output_filter, strides, block_name):
    "Bottleneck block"
    with tf.name_scope(block_name):
        result = Conv2D(filters=input_filter,
                        kernel_size=1,
                        strides=strides)(input_tensor)# padding='same'
        result = BatchNormalization()(result)
        result = Activation(ACTIVATION)(result)

        projection = Conv2D(filters=output_filter,
                            kernel_size=1,
                            strides=strides)(input_tensor)# padding='same'

        result = Conv2D(filters=input_filter, kernel_size=3,
                        padding='same')(result)
        result = BatchNormalization()(result)
        result = Activation(ACTIVATION)(result)

        result = Conv2D(filters=output_filter, kernel_size=1)(result)#padding='same'
        result = BatchNormalization()(result)
        result = tf.keras.layers.add([result, projection])
        result = Activation(ACTIVATION)(result)

    return result


def get_incoming_shape(incoming):
    """ Returns the incoming data shape """
    if isinstance(incoming, tf.Tensor):
        return incoming.get_shape().as_list()
    # elif type(incoming) in [np.array, list, tuple]:
    #     return np.shape(incoming)
    else:
        raise Exception("Invalid incoming layer.")


def interleave(tensors, axis):
    "interleaving the elements of a list of feature maps"
    old_shape = get_incoming_shape(tensors[0])[1:]
    new_shape = [-1] + old_shape
    new_shape[axis] *= len(tensors)
    return tf.reshape(tf.stack(tensors, axis + 1), new_shape)


def unpool_as_conv(input_tensor, output_filter, activation):
    "Figure 3 in paper[1]↑"
    result_A = Conv2D(filters=output_filter,
                      kernel_size=(3, 3),
                      strides=1,
                      padding='SAME')(input_tensor)
    result_B = Conv2D(filters=output_filter,
                      kernel_size=(2, 3),
                      strides=1,
                      padding='SAME')(input_tensor)
    result_C = Conv2D(filters=output_filter,
                      kernel_size=(3, 2),
                      strides=1,
                      padding='SAME')(input_tensor)
    result_D = Conv2D(filters=output_filter,
                      kernel_size=(2, 2),
                      strides=1,
                      padding='SAME')(input_tensor)

    # Interleaving elements of the four feature maps
    # --------------------------------------------------
    left = interleave([result_A, result_B], axis=1)  # columns
    right = interleave([result_C, result_D], axis=1)  # columns
    result = interleave([left, right], axis=2)  # rows

    result = BatchNormalization()(result)

    if activation:
        result = Activation(ACTIVATION)(result)

    return result


def up_project(input_tensor, output_filter, block_name):
    "Figure 2, (d) in paper[1]↑"
    with tf.name_scope(block_name):
        result_1 = unpool_as_conv(input_tensor=input_tensor,
                                  output_filter=output_filter,
                                  activation=True)

        result_1 = Conv2D(filters=output_filter,
                          kernel_size=3,
                          strides=1,
                          padding='SAME')(result_1)
        result_1 = BatchNormalization()(result_1)

        result_2 = unpool_as_conv(input_tensor=input_tensor,
                                  output_filter=output_filter,
                                  activation=False)

        result = tf.add_n([result_1, result_2])
        result = Activation(ACTIVATION)(result)

    return result


def model(img_shape=(480, 640, 3)):
    """FCRN model"""

    # #         self.features = tf.Variable(file.root.images[self.index:self.index+self.batch_size], tf.float32)
    # #         self.features = tf.transpose(self.features,perm=[0,3,2,1])
    # features = tf.to_float(features)
    # features = tf.div(features, 255.0)
    # #TODO: delete
    # features = tf.image.resize_images(features, [228, 304])

    # #         self.labels = tf.Variable(file.root.depths[self.index:self.index+self.batch_size], tf.float32)
    # #         self.labels = tf.transpose(self.labels,perm=[0,2,1])
    # labels = tf.reshape(labels, labels.get_shape().as_list() + [1])
    # #TODO: delete
    # labels = tf.image.resize_images(labels, [128, 160])

    ##         plt.figure(figsize=(10, 15))
    ##         for i in range(self.batch_size):
    ##             plt.subplot(batch_size,2,2*i+1)
    ##             plt.imshow(np.transpose(file.root.depths[i+self.index],[1,0]))
    ##             plt.subplot(batch_size,2,2*i+2)
    ##             plt.imshow(np.transpose(file.root.images[i+self.index],[2,1,0]))
    ## Input Layer
    ## Reshape X to 4-D tensor: [batch_size, width, height, channels]
    ## SYNTHIA-SF images are 1920x1080 pixels, and have three color channel

    ##     input_layer = tf.reshape(features["x"], [-1, 1920, 1080, 3])

    # input_layer = tf.reshape(features, [-1, 228, 304, 3], name='Input')
    inputs = Input(shape=img_shape)

    result = Conv2D(filters=64, kernel_size=7, strides=2,
                    padding='same')(inputs)
    result = BatchNormalization()(result)
    result = MaxPool2D(pool_size=3, strides=2, padding='same')(result)
    result = Activation(ACTIVATION)(result)

    result = bottleneck(input_tensor=result,
                        input_filter=64,
                        output_filter=256,
                        strides=1,
                        block_name='block')

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='block')

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='block')

    result = bottleneck(input_tensor=result,
                        input_filter=128,
                        output_filter=512,
                        strides=2,
                        block_name='block')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='block')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='block')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='block')

    result = bottleneck(input_tensor=result,
                        input_filter=256,
                        output_filter=1024,
                        strides=2,
                        block_name='block')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='block')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='block')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='block')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='block')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='block')

    result = bottleneck(input_tensor=result,
                        input_filter=512,
                        output_filter=2048,
                        strides=2,
                        block_name='block')

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='block')

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='block')

    result = Conv2D(filters=1024, kernel_size=1, strides=1)(result)
    result = BatchNormalization()(result)

    result = up_project(input_tensor=result,
                        output_filter=512,
                        block_name='block')
    result = up_project(input_tensor=result,
                        output_filter=256,
                        block_name='block')
    result = up_project(input_tensor=result,
                        output_filter=128,
                        block_name='block')
    result = up_project(input_tensor=result,
                        output_filter=64,
                        block_name='block')

    ####TODO: remove? (not present in the paper)
    # result = up_project(input_tensor=result,
    #                     output_filter=32,
    #                     block_name='block')
    ###

    #         result = tf.layers.dropout(inputs=result, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    outputs = Conv2D(filters=1, kernel_size=1, strides=1,
                     name='Prediction')(result)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name='FCRN')
