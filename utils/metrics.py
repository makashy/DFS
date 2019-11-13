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
        score = (intersection +
                 smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) -
                            intersection + smooth)
        return score


def class_IOU_list(class_list):
    IOU_list = []
    # class_indices = np.arange(len(class_list))
    for i, class_name in enumerate(class_list):
        IOU_list.append(classIOU(class_name, i))
    return IOU_list


################################ depth estimation ###


class MAPE(tf.keras.losses.Loss):
    """Computes the Mean Absolute Percentage Error between y_true and y_pred (percent).
    (absErrorRel in kitti dataset)
    Alternatives:
        tf.keras.metrics.MeanAbsolutePercentageError
        tf.keras.losses.MeanAbsolutePercentageError

    Arguments:
        smooth: For avoiding devison by zero
        minimum_y_true: minimum acceptable value for y_true.
            For those y_true smaller than minimum_y_true, the result is zero.
    """
    def __init__(self,
                 smooth=1e-6,
                 minimum_y_true=0.5,
                 reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction, name='MAPE')
        self.smooth = smooth
        self.minimum_y_true = minimum_y_true

    def call(self, y_true, y_pred):
        "Computes and returns the metric value tensor."
        return tf.reduce_mean(
            tf.abs(y_true - y_pred) *
            tf.cast(tf.logical_not(y_true < self.minimum_y_true), tf.float32) /
            (y_true + self.smooth) * 100)


class MSPE(tf.keras.losses.Loss):
    """Computes the Mean Square Percentage Error between y_true and y_pred (percent).
    (absErrorRel in kitti dataset)
    Alternatives:
        tf.keras.metrics.MeanAbsolutePercentageError
        tf.keras.losses.MeanAbsolutePercentageError

    Arguments:
        smooth: For avoiding devison by zero
        minimum_y_true: minimum acceptable value for y_true.
            For those y_true smaller than minimum_y_true, the result is zero
    """
    def __init__(self,
                 smooth=1e-6,
                 minimum_y_true=0.5,
                 reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction, name='MSPE')
        self.smooth = smooth
        self.minimum_y_true = minimum_y_true

    def call(self, y_true, y_pred):
        "Computes and returns the metric value tensor."
        return tf.reduce_mean(
            tf.pow(y_true - y_pred, 2) *
            tf.cast(tf.logical_not(y_true < self.minimum_y_true), tf.float32) /
            (y_true + self.smooth) * 100)


class RMSE(tf.keras.losses.Loss):
    """Computes root mean squared error metric between y_true and y_pred.
    Alternatives:
        tf.keras.metrics.RootMeanSquaredError
    """
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction, name='RMSE')

    def call(self, y_true, y_pred):
        "Computes and returns the metric value tensor."
        return tf.sqrt(tf.reduce_mean(tf.pow(y_true - y_pred, 2)))


class RMSELog(tf.keras.losses.Loss):
    """Computes root mean squared error metric in log space  between y_true and y_pred.
    """
    def __init__(self, smooth=1e-6, reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction, name='RMSELog')
        self.smooth = smooth

    def call(self, y_true, y_pred):
        "Computes and returns the metric value tensor."
        return tf.sqrt(
            tf.reduce_mean(
                tf.pow(
                    tf.math.log(y_true + self.smooth) -
                    tf.math.log(y_pred + self.smooth), 2)))


class Log10(tf.keras.losses.Loss):
    """Computes mean absolute error metric in log10 space between y_true and y_pred.
    """
    def __init__(self, smooth=1e-6, reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction, name='Log10')
        self.smooth = smooth

    def call(self, y_true, y_pred):
        "Computes and returns the metric value tensor."
        return tf.reduce_mean(
            tf.abs(
                tf.math.log(y_true + self.smooth) -
                tf.math.log(y_pred + self.smooth)))


class DeltaThreshold(tf.keras.losses.Loss):
    """Computes delta threshold metric in between y_true and y_pred.
    """
    def __init__(self,
                 smooth=1e-6,
                 i=1,
                 reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction, name='delta_threshold_' + str(i))
        self.i = i
        self.smooth = smooth

    def call(self, y_true, y_pred):
        "Computes and returns the metric value tensor."
        return tf.math.count_nonzero(
            tf.maximum(y_true / (self.smooth + y_pred), y_pred /
                       (self.smooth + y_true)) < 1.25**self.i) / tf.size(
                           y_true, out_type=tf.dtypes.int64)


class SILog(tf.keras.losses.Loss):
    """Computes  Scale invariant logarithmic error metric in between y_true and y_pred.
    """
    def __init__(self, smooth=1e-6, reduction=tf.keras.losses.Reduction.AUTO):
        super().__init__(reduction=reduction, name='SILog')
        self.smooth = smooth

    def call(self, y_true, y_pred):
        "Computes and returns the metric value tensor."
        difference = tf.math.log(y_true + self.smooth) - \
                     tf.math.log(y_pred + self.smooth)
        return tf.reduce_mean(tf.pow(difference, 2)) - tf.pow(
            tf.reduce_mean(difference), 2)
