"""Contains dataset importers for NYU Depth Dataset V2 and SYNTHIA-SF"""

from __future__ import absolute_import, division, print_function

import os
import pandas as pd
import numpy as np
import tables
import imageio


class NYU:
    """Iterator for looping over elements of NYU Depth Dataset V2 backwards."""

    def __init__(self,
                 NYU_Depth_Dataset_V2_address,
                 batch_size=1,
                 repeater=False,
                 label_type='segmentation'):
        self.file = tables.open_file(NYU_Depth_Dataset_V2_address)
        self.index = len(self.file.root.images) - 1
        self.batch_size = batch_size
        self.repeater = repeater
        if isinstance(label_type, str):
            self.label_type = label_type
        else:
            self.label_type = label_type.decode()

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        """Retrieve the next pairs from the dataset
        """
        if self.index - self.batch_size <= 0:
            if not self.repeater:
                raise StopIteration
            else:
                self.index = len(self.file.root.images) - 1
        self.index = self.index - self.batch_size

        features = np.array(
            self.file.root.images[self.index:self.index + self.batch_size])
        features = np.transpose(features, [0, 3, 2, 1])

        if self.label_type == 'depth':
            label_set = self.file.root.depths
            labels = np.array(
                label_set[self.index:self.index + self.batch_size])
            labels = np.transpose(labels, [0, 2, 1])
            labels = np.reshape(
                labels, (labels.shape[0], labels.shape[1], labels.shape[2], 1))
        elif self.label_type == 'segmentation':
            label_set = self.file.root.labels
            labels = np.array(
                label_set[self.index:self.index + self.batch_size])
            labels = np.transpose(labels, [0, 2, 1])
            labels = np.reshape(
                labels, (labels.shape[0], labels.shape[1], labels.shape[2], 1))
            new_labels = np.ndarray(shape=(self.batch_size, 480, 640, 21))

            for i in range(self.batch_size):
                for j in range(480):
                    for k in range(640):
                        if labels[i, j, k] < 21:
                            new_labels[i, j, k, int(labels[i, j, k])] = 1
            labels = new_labels
        else:
            raise ValueError('invalid label type')

        return features, labels


class SynthiaSf:
    """Iterator for looping over elements of SYNTHIA-SF backwards."""

    def __init__(self,
                 synthia_sf_address,
                 batch_size=1,
                 repeater=False,
                 label_type='segmentation',
                 shuffle=False):

        self.dataset_address = synthia_sf_address
        self.sequence_folder = [
            '/SEQ1', '/SEQ2', '/SEQ3', '/SEQ4', '/SEQ5', '/SEQ6'
        ]
        self.rgb_folder = ['/RGBLeft/', '/RGBRight/']
        self.depth_folder = ['/DepthLeft/', '/DepthRight/']
        self.segmentation_folder = ['/GTLeft/', '/GTright/']
        self.batch_size = batch_size
        self.repeater = repeater
        self.label_type = label_type
        self.shuffle = shuffle
        self.dataset = self.data_frame_creator()
        self.index = self.dataset.shape[0] - 1

    def data_frame_creator(self):
        """ pandas dataFrame for addresses of rgb, depth and segmentation"""

        rgb_dir = [
            self.dataset_address + sequence_f + rgb_f
            for rgb_f in self.rgb_folder for sequence_f in self.sequence_folder
        ]
        rgb_data = [
            rgb_d + rgb for rgb_d in rgb_dir for rgb in os.listdir(rgb_d)
        ]

        depth_dir = [
            self.dataset_address + sequence_f + depth_f
            for depth_f in self.depth_folder
            for sequence_f in self.sequence_folder
        ]
        depth_data = [
            depth_d + depth for depth_d in depth_dir
            for depth in os.listdir(depth_d)
        ]

        segmentation_dir = [
            self.dataset_address + sequence_f + segmentation_f
            for segmentation_f in self.segmentation_folder
            for sequence_f in self.sequence_folder
        ]
        segmentation_data = [
            segmentation_d + segmentation
            for segmentation_d in segmentation_dir
            for segmentation in os.listdir(segmentation_d)
        ]

        dataset = {
            'RGB': rgb_data,
            'DEPTH': depth_data,
            'SEGMENTATION': segmentation_data
        }

        if self.shuffle:
            return pd.DataFrame(dataset).sample(frac=1)

        return pd.DataFrame(dataset)

    def __iter__(self):
        return self

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def next(self):
        """Retrieve the next pairs from the dataset"""

        if self.index - self.batch_size <= 0:
            if not self.repeater:
                raise StopIteration
            else:
                self.index = self.dataset.shape[0] - 1
        self.index = self.index - self.batch_size

        features = np.array(
            np.array([
                imageio.imread(self.dataset.iloc[i, 0])
                for i in range(self.index, self.index + self.batch_size)
            ]))

        if self.label_type == 'depth':
            labels = np.array(
                np.array([
                    imageio.imread(self.dataset.iloc[i, 1])
                    for i in range(self.index, self.index + self.batch_size)
                ]),
                dtype=np.int32)
            labels = (labels[:, :, :, 0] + labels[:, :, :, 1] * 256 +
                      labels[:, :, :, 2] * 256 * 256) / ((256 * 256 * 256) - 1)

        elif self.label_type == 'segmentation':
            labels = np.array(
                np.array([
                    imageio.imread(self.dataset.iloc[i, 2])[:, :, 0]
                    for i in range(self.index, self.index + self.batch_size)
                ]))
        else:
            raise ValueError('invalid label type')

        return features, labels
