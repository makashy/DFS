"""Contains metrics for evaluating segmentation DNN """
import numpy as np
from numba import jit
import tensorflow as tf


def mean_IOU(y_true, y_pred):
    "Calculates mean intersection over uninon over all classes"
    smooth = 1e-6
    y_true_f = tf.keras.backend.reshape(y_true, [-1])
    y_pred_f = tf.keras.backend.reshape(y_pred, [-1])
    true_positive = tf.keras.backend.sum(y_true_f * y_pred_f)
    score = (true_positive + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) -
        true_positive + smooth)  # = TP/(TP + FP + FN)
    return score


def mean_IOU_loss(y_true, y_pred):
    "Calculates IOU loss"
    loss = 1 - mean_IOU(y_true, y_pred)
    return loss


############################################################tensorflow : bce_dice_loss
def dice_coeff(y_true, y_pred):
    ""
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) +
                                            tf.reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(
        y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss


############################################################numpy : bce_dice_loss
# numpy methodes have some problems with keras training procedure.
def dice_coeff_np(y_true, y_pred):
    ""
    smooth = 1.
    y_true_f = np.reshape(y_true, [-1])
    y_pred_f = np.reshape(y_pred, [-1])
    intersection = np.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (np.sum(y_true_f) +
                                            np.sum(y_pred_f) + smooth)
    return score


def dice_loss_np(y_true, y_pred):
    loss = 1 - dice_coeff_np(y_true, y_pred)
    return loss


def bce_dice_loss_np(y_true, y_pred):
    loss =  tf.cast(tf.numpy_function(dice_loss_np, [y_true, y_pred], tf.float64), tf.float32) + \
    tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return loss


############################################################numba : bce_dice_loss
# numba methodes have some problems with keras training procedure.
@jit
def dice_coeff_nb(y_true, y_pred):
    ""
    smooth = np.float32(1)
    y_true_f = np.reshape(y_true, [-1])
    y_pred_f = np.reshape(y_pred, [-1])
    intersection = np.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (np.sum(y_true_f) +
                                            np.sum(y_pred_f) + smooth)
    return score


@jit
def dice_loss_nb(y_true, y_pred):
    loss = 1 - dice_coeff_nb(y_true, y_pred)
    return loss


def bce_dice_loss_nb(y_true, y_pred):
    loss =  tf.cast(tf.numpy_function(dice_loss_nb, [y_true, y_pred], tf.float64), tf.float32) + \
    tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return loss

############################################################tensorflow : IOU class
class classIOU:
    def __init__(self, name, index):
        self.__name__ = name
        self.index = index

    def __call__(self, y_true, y_pred):
        ""
        smooth = 1e-6
        y_true_f = tf.reshape(y_true[:, :, :, self.index], [-1])
        y_pred_f = tf.reshape(y_pred[:, :, :, self.index], [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        score = (intersection + smooth) / (tf.reduce_sum(y_true_f) +
                                           tf.reduce_sum(y_pred_f) - intersection + smooth)
        return score

def class_IOU_list(class_list):
    IOU_list = []
    # class_indices = np.arange(len(class_list))
    for i, class_name in enumerate(class_list):
        IOU_list.append(classIOU(class_name, i))
    return IOU_list
