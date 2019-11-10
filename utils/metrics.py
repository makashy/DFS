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


################################ depth estimation ###


def MAPE_cupy(y_true, y_pred, smooth=1e-6):
    """Computes the Mean Absolute Percentage Error between y_true and y_pred (percent).
    (absErrorRel in kitti dataset)
    Alternatives:
        tf.keras.metrics.MeanAbsolutePercentageError
        tf.keras.losses.MeanAbsolutePercentageError
    """
    y_t = cp.array(y_true)
    y_p = cp.array(y_pred)
    return cp.mean(cp.abs(y_t - y_p) / (y_t + smooth) * 100)


def RMSE_cupy(y_true, y_pred):
    """Computes root mean squared error metric between y_true and y_pred.
    Alternatives:
        tf.keras.metrics.RootMeanSquaredError
    """
    y_t = cp.array(y_true)
    y_p = cp.array(y_pred)
    return cp.sqrt(cp.mean(cp.power(y_t - y_p, 2)))


def RMSElog_cupy(y_true, y_pred, smooth=1e-6):
    """Computes root mean squared error metric in log space  between y_true and y_pred.
    """
    y_t = cp.array(y_true)
    y_p = cp.array(y_pred)
    return cp.sqrt(cp.mean(cp.power(cp.log(y_t + smooth) - cp.log(y_p + smooth), 2)))


def log10_cupy(y_true, y_pred, smooth=1e-6):
    """Computes mean absolute error metric in log10 space between y_true and y_pred.
    """
    y_t = cp.array(y_true)
    y_p = cp.array(y_pred)
    return cp.mean(cp.abs(cp.log10(y_t + smooth) - cp.log10(y_p + smooth)))


def delta_threshold_cupy(y_true, y_pred, smooth=1e-6, i=1):
    """Computes delta threshold metric in between y_true and y_pred.
    """
    y_t = cp.array(y_true)
    y_p = cp.array(y_pred)
    return cp.count_nonzero(cp.maximum(y_t/(smooth + y_p), y_p/(smooth + y_t)) < 1.25**i)/y_t.size


def SILog_cupy(y_true, y_pred):
    """Computes  Scale invariant logarithmic error metric in between y_true and y_pred.
    """
    y_t = cp.array(y_true)
    y_p = cp.array(y_pred)
    difference = cp.log(y_t + smooth) - cp.log(y_p + smooth)
    return cp.mean(cp.power(difference, 2)) -  cp.power(cp.mean(difference), 2)


# absErrorRel:  Relative absolute error (percent)

# iRMSE:  Root mean squared error of the inverse depth [1/km]