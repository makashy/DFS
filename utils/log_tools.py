""" Contains tools for store network performance logs during training
"""

import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf


class VisualLogger(tf.keras.callbacks.Callback):
    """Callback that examines the model against validation images.

    This callback uses a DatasetGenerator to generate a batch of
    features and labels. On epoch ends it stores model prediction
    to track its progress.

    Arguments:
        log_dir: the path of the directory where to save the log file
        validation_dataset: a DatasetGenerator object.
        index: index of the batch to be used for examining the model
    """
    def __init__(self, log_dir, validation_dataset, index=0):
        super().__init__()
        self.log_dir = log_dir

        if 'visual_log.pkl' in os.listdir(log_dir):
            self.result_dataframe = pd.read_pickle(log_dir + "/visual_log.pkl")
        else:
            self.result_dataframe = pd.DataFrame()

            validation_dataset.index = index
            preview = iter(validation_dataset)
            self.feature_list, self.label_list = next(preview)

            self.result_dataframe['feature_list'] = [self.feature_list]
            self.result_dataframe['label_list'] = [self.label_list]

    def on_epoch_end(self, epoch, logs):
        """Called at the end of an epoch.
           Stores visual_log.pkl on log_dir.
        """
        predict = self.model.predict(self.feature_list, steps=1)
        self.result_dataframe['predict ' + str(epoch)] = [predict]
        self.result_dataframe.to_pickle(self.log_dir + 'visual_log.pkl')


class MetricLogger(tf.keras.callbacks.Callback):
    """Callback that streams epoch results to a pkl file.

    Arguments:
        log_dir: the path of the directory where to save the log file
    """
    def __init__(self, log_dir):
        super().__init__()
        self.log_dir = log_dir

        if 'metric_log.pkl' in os.listdir(log_dir):
            self.metric_dataframe = pd.read_pickle(log_dir + "/metric_log.pkl")
        else:
            self.metric_dataframe = pd.DataFrame()

    def on_epoch_end(self, epoch, logs):
        """Called at the end of an epoch.
           Stores metric_log.pkl on log_dir.
        """
        if self.metric_dataframe.columns.size == 0:
            self.metric_dataframe = pd.DataFrame(columns=[
                'epoch',
                'time',
            ] + list(logs.keys()))
        self.metric_dataframe.loc[epoch] = np.array([
            epoch,
            time.time(),
        ] + list(logs.values()),
                                                    dtype=np.float64)
        self.metric_dataframe.to_pickle(self.log_dir + 'metric_log.pkl')
