"""FCRN model:
[1] I. Laina, C. Rupprecht, V. Belagiannis, F. Tombari, and N. Navab,
“Deeper Depth Prediction with Fully Convolutional Residual Networks,”
arXiv:1606.00373 [cs], Jun. 2016.
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras.layers import (  # pylint: disable=import-error
    Activation, BatchNormalization, Conv2D, Input, MaxPool2D, Concatenate,
    Multiply, Add)

ACTIVATION = 'relu'  #'selu'


def baseline(input_tensor,
             input_filter,
             output_filter,
             block_name,
             batch_normalization=True):
    "Baseline block"
    with tf.name_scope(block_name):
        result = Conv2D(filters=input_filter, kernel_size=1,
                        padding='same')(input_tensor)
        if batch_normalization:
            result = BatchNormalization()(result)
        result = Activation(ACTIVATION)(result)

        result = Conv2D(filters=input_filter, kernel_size=3,
                        padding='same')(input_tensor)
        if batch_normalization:
            result = BatchNormalization()(result)
        result = Activation(ACTIVATION)(result)

        result = Conv2D(filters=output_filter, kernel_size=1,
                        padding='same')(input_tensor)
        if batch_normalization:
            result = BatchNormalization()(result)
        result = tf.keras.layers.add([result, input_tensor])
        result = Activation(ACTIVATION)(result)

    return result


def bottleneck(input_tensor,
               input_filter,
               output_filter,
               strides,
               block_name,
               batch_normalization=True):
    "Bottleneck block"
    with tf.name_scope(block_name):
        result = Conv2D(filters=input_filter, kernel_size=1,
                        strides=strides)(input_tensor)  # padding='same'
        if batch_normalization:
            result = BatchNormalization()(result)
        result = Activation(ACTIVATION)(result)

        projection = Conv2D(filters=output_filter,
                            kernel_size=1,
                            strides=strides)(input_tensor)  # padding='same'

        result = Conv2D(filters=input_filter, kernel_size=3,
                        padding='same')(result)
        if batch_normalization:
            result = BatchNormalization()(result)
        result = Activation(ACTIVATION)(result)

        result = Conv2D(filters=output_filter,
                        kernel_size=1)(result)  #padding='same'
        if batch_normalization:
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
    return tf.keras.backend.reshape(tf.keras.backend.stack(tensors, axis + 1),
                                    new_shape)


def unpool_as_conv(input_tensor,
                   output_filter,
                   activation,
                   name,
                   batch_normalization=True):
    "Figure 3 in paper[1]↑"
    with tf.name_scope(name):
        result_A = Conv2D(filters=output_filter,
                          kernel_size=(3, 3),
                          strides=1,
                          padding='SAME',
                          name=tf.compat.v1.get_default_graph().unique_name(
                              'result_A'))(input_tensor)
        result_B = Conv2D(filters=output_filter,
                          kernel_size=(2, 3),
                          strides=1,
                          padding='SAME',
                          name=tf.compat.v1.get_default_graph().unique_name(
                              'result_B'))(input_tensor)
        result_C = Conv2D(filters=output_filter,
                          kernel_size=(3, 2),
                          strides=1,
                          padding='SAME',
                          name=tf.compat.v1.get_default_graph().unique_name(
                              'result_C'))(input_tensor)
        result_D = Conv2D(filters=output_filter,
                          kernel_size=(2, 2),
                          strides=1,
                          padding='SAME',
                          name=tf.compat.v1.get_default_graph().unique_name(
                              'result_D'))(input_tensor)

        # Interleaving elements of the four feature maps
        # --------------------------------------------------
        left = interleave([result_A, result_B], axis=1)  # columns
        right = interleave([result_C, result_D], axis=1)  # columns
        result = interleave([left, right], axis=2)  # rows
        if batch_normalization:
            result = BatchNormalization()(result)

        if activation:
            result = Activation(ACTIVATION)(result)

    return result


def up_project(input_tensor,
               output_filter,
               block_name,
               batch_normalization=True):
    "Figure 2, (d) in paper[1]↑"
    with tf.name_scope(block_name):
        result_1 = unpool_as_conv(input_tensor=input_tensor,
                                  output_filter=output_filter,
                                  activation=True,
                                  name='result_1',
                                  batch_normalization=batch_normalization)

        result_1 = Conv2D(filters=output_filter,
                          kernel_size=3,
                          strides=1,
                          padding='SAME')(result_1)
        if batch_normalization:
            result_1 = BatchNormalization()(result_1)

        result_2 = unpool_as_conv(input_tensor=input_tensor,
                                  output_filter=output_filter,
                                  activation=False,
                                  name='result_2',
                                  batch_normalization=batch_normalization)

        result = tf.keras.layers.add([result_1, result_2])
        result = Activation(ACTIVATION)(result)

    return result


## M0: original FCRN (With addition of a up_project at last layers
# and additional con2d at final layer)
def model_m0(rgb_shape=(480, 640, 3)):
    """FCRN model"""

    inputs = Input(shape=rgb_shape)

    result = Conv2D(filters=64, kernel_size=7, strides=2,
                    padding='same')(inputs)
    result = BatchNormalization()(result)
    result = MaxPool2D(pool_size=3, strides=2, padding='same')(result)
    result = Activation(ACTIVATION)(result)

    result = bottleneck(input_tensor=result,
                        input_filter=64,
                        output_filter=256,
                        strides=1,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=128,
                        output_filter=512,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=256,
                        output_filter=1024,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=512,
                        output_filter=2048,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='baseline')

    result = Conv2D(filters=1024, kernel_size=1, strides=1)(result)
    result = BatchNormalization()(result)

    result = up_project(input_tensor=result,
                        output_filter=512,
                        block_name='up_project')
    result = up_project(input_tensor=result,
                        output_filter=256,
                        block_name='up_project')
    result = up_project(input_tensor=result,
                        output_filter=128,
                        block_name='up_project')
    result = up_project(input_tensor=result,
                        output_filter=64,
                        block_name='up_project')
    result = up_project(input_tensor=result,
                        output_filter=32,
                        block_name='up_project_additional')

    # result = tf.layers.dropout(inputs=result, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    outputs = Conv2D(filters=1, kernel_size=1, strides=1,
                     name='predict')(result)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name='FCRN')


## M1: It only use a concatenated input with combination of RGB image and Semantic Segmentation Map
def model_m1(rgb_shape=(480, 640, 3),
             seg_shape=(480, 640, 19),
             batch_normalization=True):
    """FCRN model"""

    inputs_rgb = Input(shape=rgb_shape)
    inputs_seg = Input(shape=seg_shape)

    result = Concatenate()([inputs_rgb, inputs_seg])
    result = Conv2D(filters=64, kernel_size=7, strides=2,
                    padding='same')(result)
    if batch_normalization:
        result = BatchNormalization()(result)
    result = MaxPool2D(pool_size=3, strides=2, padding='same')(result)
    result = Activation(ACTIVATION)(result)

    result = bottleneck(input_tensor=result,
                        input_filter=64,
                        output_filter=256,
                        strides=1,
                        block_name='bottleneck',
                        batch_normalization=batch_normalization)

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='baseline',
                      batch_normalization=batch_normalization)

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='baseline',
                      batch_normalization=batch_normalization)

    result = bottleneck(input_tensor=result,
                        input_filter=128,
                        output_filter=512,
                        strides=2,
                        block_name='bottleneck',
                        batch_normalization=batch_normalization)

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline',
                      batch_normalization=batch_normalization)

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline',
                      batch_normalization=batch_normalization)

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline',
                      batch_normalization=batch_normalization)

    result = bottleneck(input_tensor=result,
                        input_filter=256,
                        output_filter=1024,
                        strides=2,
                        block_name='bottleneck',
                        batch_normalization=batch_normalization)

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline',
                      batch_normalization=batch_normalization)

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline',
                      batch_normalization=batch_normalization)

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline',
                      batch_normalization=batch_normalization)

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline',
                      batch_normalization=batch_normalization)

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline',
                      batch_normalization=batch_normalization)

    result = bottleneck(input_tensor=result,
                        input_filter=512,
                        output_filter=2048,
                        strides=2,
                        block_name='bottleneck',
                        batch_normalization=batch_normalization)

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='baseline',
                      batch_normalization=batch_normalization)

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='baseline',
                      batch_normalization=batch_normalization)

    result = Conv2D(filters=1024, kernel_size=1, strides=1)(result)
    if batch_normalization:
        result = BatchNormalization()(result)

    result = up_project(input_tensor=result,
                        output_filter=512,
                        block_name='up_project',
                        batch_normalization=batch_normalization)
    result = up_project(input_tensor=result,
                        output_filter=256,
                        block_name='up_project',
                        batch_normalization=batch_normalization)
    result = up_project(input_tensor=result,
                        output_filter=128,
                        block_name='up_project',
                        batch_normalization=batch_normalization)
    result = up_project(input_tensor=result,
                        output_filter=64,
                        block_name='up_project',
                        batch_normalization=batch_normalization)

    result = up_project(input_tensor=result,
                        output_filter=32,
                        block_name='up_project_additional',
                        batch_normalization=batch_normalization)

    result = Conv2D(filters=1, kernel_size=1, strides=1)(result)

    outputs = Activation('relu', name='Prediction')(result)

    return tf.keras.models.Model(inputs=[inputs_rgb, inputs_seg],
                                 outputs=outputs,
                                 name='FCRN')


## M2: It has additional decoder branch for each semantic class
def model_m2(rgb_shape=(480, 640, 3), seg_shape=(480, 640, 19)):
    """FCRN model"""

    inputs_rgb = Input(shape=rgb_shape)
    inputs_seg = Input(shape=seg_shape)

    result = Concatenate()([inputs_rgb, inputs_seg])
    result = Conv2D(filters=64, kernel_size=7, strides=2,
                    padding='same')(result)
    result = BatchNormalization()(result)
    result = MaxPool2D(pool_size=3, strides=2, padding='same')(result)
    result = Activation(ACTIVATION)(result)

    result = bottleneck(input_tensor=result,
                        input_filter=64,
                        output_filter=256,
                        strides=1,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=128,
                        output_filter=512,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=256,
                        output_filter=1024,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=512,
                        output_filter=2048,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='baseline')

    result = Conv2D(filters=1024, kernel_size=1, strides=1)(result)
    result = BatchNormalization()(result)

    result = up_project(input_tensor=result,
                        output_filter=512,
                        block_name='up_project')
    result = up_project(input_tensor=result,
                        output_filter=256,
                        block_name='up_project')
    result = up_project(input_tensor=result,
                        output_filter=128,
                        block_name='up_project')

    semantic_depth = []
    for _ in range(seg_shape[2]):
        temp = Conv2D(filters=4,
                      kernel_size=3,
                      padding='same',
                      name=tf.compat.v1.get_default_graph().unique_name(
                          'seperator'))(result)
        temp = up_project(input_tensor=temp,
                          output_filter=2,
                          block_name='up_project')
        temp = up_project(input_tensor=temp,
                          output_filter=1,
                          block_name='up_project')
        semantic_depth.append(temp)

    complete_depth_map = Add(name='complete_depth_map')(semantic_depth)
    semantic_depth_maps = Concatenate(
        name='semantic_depth_maps')(semantic_depth)

    return tf.keras.models.Model(
        inputs=[inputs_rgb, inputs_seg],
        outputs=[complete_depth_map, semantic_depth_maps],
        name='FCRN')


## M3: It has additional decoder branch for each semantic class
def model_m3(rgb_shape=(480, 640, 3), seg_shape=(480, 640, 19)):
    """FCRN model"""

    inputs_rgb = Input(shape=rgb_shape)
    inputs_seg = Input(shape=seg_shape)

    result = Concatenate()([inputs_rgb, inputs_seg])
    result = Conv2D(filters=64, kernel_size=7, strides=2,
                    padding='same')(result)
    result = BatchNormalization()(result)
    result = MaxPool2D(pool_size=3, strides=2, padding='same')(result)
    result = Activation(ACTIVATION)(result)

    result = bottleneck(input_tensor=result,
                        input_filter=64,
                        output_filter=256,
                        strides=1,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=128,
                        output_filter=512,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=256,
                        output_filter=1024,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=512,
                        output_filter=2048,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='baseline')

    result = Conv2D(filters=1024, kernel_size=1, strides=1)(result)
    result = BatchNormalization()(result)

    result = up_project(input_tensor=result,
                        output_filter=512,
                        block_name='up_project')
    result = up_project(input_tensor=result,
                        output_filter=256,
                        block_name='up_project')
    result = up_project(input_tensor=result,
                        output_filter=128,
                        block_name='up_project')

    semantic_depth = []
    for _ in range(seg_shape[2]):
        temp = Conv2D(filters=4,
                      kernel_size=3,
                      padding='same',
                      name=tf.compat.v1.get_default_graph().unique_name(
                          'seperator'))(result)
        temp = up_project(input_tensor=temp,
                          output_filter=2,
                          block_name='up_project')
        temp = up_project(input_tensor=temp,
                          output_filter=1,
                          block_name='up_project')
        semantic_depth.append(temp)

    predict = Add()(semantic_depth)
    predict = Activation('relu', name='predict')(predict)
    outputs = [predict]
    for class_output in semantic_depth:
        outputs.append(Activation('relu')(class_output))

    return tf.keras.models.Model(inputs=[inputs_rgb, inputs_seg],
                                 outputs=outputs,
                                 name='FCRN')


def model_m4(rgb_shape=(480, 640, 3), seg_shape=(480, 640, 19)):
    """FCRN model"""

    inputs_rgb = Input(shape=rgb_shape)
    inputs_seg = Input(shape=seg_shape)

    result = Concatenate()([inputs_rgb, inputs_seg])
    result = Conv2D(filters=64, kernel_size=7, strides=2,
                    padding='same')(result)
    result = BatchNormalization()(result)
    result = MaxPool2D(pool_size=3, strides=2, padding='same')(result)
    result = Activation(ACTIVATION)(result)

    result = bottleneck(input_tensor=result,
                        input_filter=64,
                        output_filter=256,
                        strides=1,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=128,
                        output_filter=512,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=256,
                        output_filter=1024,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=512,
                        output_filter=2048,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='baseline')

    result = Conv2D(filters=1024, kernel_size=1, strides=1)(result)
    result = BatchNormalization()(result)

    # result = up_project(input_tensor=result,
    #                     output_filter=512,
    #                     block_name='up_project')
    # result = up_project(input_tensor=result,
    #                     output_filter=256,
    #                     block_name='up_project')
    # result = up_project(input_tensor=result,
    #                     output_filter=128,
    #                     block_name='up_project')
    # result = up_project(input_tensor=result,
    #                     output_filter=64,
    #                     block_name='up_project')

    # result = up_project(input_tensor=result,
    #                     output_filter=32,
    #                     block_name='up_project_additional')

    semantic_depth = []
    for i in range(seg_shape[2]):
        with tf.name_scope('semantic_depth_layer_' + str(i)):
            print('semantic_depth_layer_' + str(i))
            sem_rgb = tf.multiply(inputs_rgb,
                                  tf.expand_dims(inputs_seg[:, :, :, i], -1))

            temp = Conv2D(filters=32,
                          kernel_size=3,
                          padding='same',
                          name=tf.compat.v1.get_default_graph().unique_name(
                              'seperator'))(result)

            temp = up_project(input_tensor=temp,
                              output_filter=16,
                              block_name='up_project')

            temp = Concatenate()([temp, tf.image.resize(sem_rgb, (30, 40))])

            temp = up_project(input_tensor=temp,
                              output_filter=8,
                              block_name='up_project')
            temp = up_project(input_tensor=temp,
                              output_filter=4,
                              block_name='up_project')
            temp = up_project(input_tensor=temp,
                              output_filter=2,
                              block_name='up_project')
            temp = up_project(input_tensor=temp,
                              output_filter=1,
                              block_name='up_project')

            # temp = Multiply()(
            #     [temp, tf.image.resize(tf.expand_dims(inputs_seg[:, :, :, i], -1), (480, 640))])
            semantic_depth.append(temp)

    predict = Add()(semantic_depth)
    predict = Activation('relu', name='predict')(predict)
    outputs = [predict]
    for class_output in semantic_depth:
        outputs.append(Activation('relu')(class_output))

    return tf.keras.models.Model(inputs=[inputs_rgb, inputs_seg],
                                 outputs=outputs,
                                 name='FCRN')


def model_m5(rgb_shape=(480, 640, 3), seg_shape=(480, 640, 19)):
    """FCRN model"""

    inputs_rgb = Input(shape=rgb_shape)
    inputs_seg = Input(shape=seg_shape)

    result = Concatenate()([inputs_rgb, inputs_seg])
    result = Conv2D(filters=64, kernel_size=7, strides=2,
                    padding='same')(result)
    result = BatchNormalization()(result)
    result = MaxPool2D(pool_size=3, strides=2, padding='same')(result)
    result = Activation(ACTIVATION)(result)

    result = bottleneck(input_tensor=result,
                        input_filter=64,
                        output_filter=256,
                        strides=1,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=64,
                      output_filter=256,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=128,
                        output_filter=512,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=128,
                      output_filter=512,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=256,
                        output_filter=1024,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=256,
                      output_filter=1024,
                      block_name='baseline')

    result = bottleneck(input_tensor=result,
                        input_filter=512,
                        output_filter=2048,
                        strides=2,
                        block_name='bottleneck')

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='baseline')

    result = baseline(input_tensor=result,
                      input_filter=512,
                      output_filter=2048,
                      block_name='baseline')

    result = Conv2D(filters=1024, kernel_size=1, strides=1)(result)
    result = BatchNormalization()(result)

    result = up_project(input_tensor=result,
                        output_filter=512,
                        block_name='up_project')
    result = up_project(input_tensor=result,
                        output_filter=256,
                        block_name='up_project')

    semantic_depth = []
    for i in range(seg_shape[2]):
        with tf.name_scope('semantic_depth_layer_' + str(i)):
            print('semantic_depth_layer_' + str(i))

            sem_rgb = Multiply()(
                [inputs_rgb,
                 tf.expand_dims(inputs_seg[:, :, :, i], -1)])

            temp = Concatenate()([result, tf.image.resize(sem_rgb, (60, 80))])

            temp = up_project(input_tensor=temp,
                              output_filter=4,
                              block_name='up_project')

            # temp = Concatenate()([temp, tf.image.resize(sem_rgb, (120, 160))])

            temp = up_project(input_tensor=temp,
                              output_filter=2,
                              block_name='up_project')
            temp = up_project(input_tensor=temp,
                              output_filter=1,
                              block_name='up_project')

            temp = Multiply()([
                temp,
                tf.image.resize(tf.expand_dims(inputs_seg[:, :, :, i], -1),
                                (480, 640))
            ])
            semantic_depth.append(temp)

    predict = Add()(semantic_depth)
    predict = Activation('relu', name='predict')(predict)
    outputs = [predict]
    for class_output in semantic_depth:
        outputs.append(Activation('relu')(class_output))

    return tf.keras.models.Model(inputs=[inputs_rgb, inputs_seg],
                                 outputs=outputs,
                                 name='FCRN')

