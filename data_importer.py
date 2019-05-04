"""Contains dataset importers for NYU Depth Dataset V2 and SYNTHIA-SF"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd
import tables
from skimage import img_as_float32
from skimage import img_as_float64
from skimage.io import imread
from skimage.transform import resize
from skimage import img_as_ubyte

import segmentation_dics


class DatasetGenerator:
    """Abstract iterator for looping over elements of a dataset ."""

    def __init__(self,
                 dataset_dir,
                 batch_size=1,
                 repeater=False,
                 shuffle=False,
                 output_shape=None,
                 data_type='float64',
                 label_type=('segmentation', 'instance', 'depth')):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.repeater = repeater
        self.shuffle = shuffle
        self.output_shape = output_shape
        self.data_type = data_type
        self.label_type = label_type
        self.dataset = self.data_frame_creator()
        self.index = self.dataset.shape[0] - 1
        self.seg_dic = {(0, 0, 0): (0, 'unlabeled')}
        self.init()

    def init(self):
        pass

    def data_frame_creator(self):
        """Pandas dataFrame for addresses of images and corresponding labels"""

        return pd.DataFrame()

    def __iter__(self):
        return self

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

        # loading features(images)
        features = imread(self.dataset.loc[:, 'image'].iat[0])[:, :, :3]

        if self.output_shape is None:
            output_shape = features.shape[:2]
        else:
            output_shape = self.output_shape

        # 1) Resize image to match a certain size.
        # 2) Also the input image is converted (from 8-bit integer)
        # to 64-bit floating point(->preserve_range=False).
        # 3) [:, :, :3] -> to remove 4th channel in png
        features = np.array([
            resize(
                image=imread(self.dataset.loc[:, 'image'].iat[i])[:, :, :3],
                output_shape=output_shape,
                mode='constant',
                preserve_range=False,
                anti_aliasing=True)
            for i in range(self.index, self.index + self.batch_size)
        ])

        if self.data_type is 'float32':
            features = img_as_float32(features)

        # loading labels(segmentation)
        if 'segmentation' in self.label_type:
            # 1) Resize segmentation to match a certain size.
            # 2) [:, :, :3] -> to remove 4th channel in png
            segmentation = np.array([
                imread(
                    self.dataset.loc[:, 'segmentation'].iat[i])[:, :, :3]
                    for i in range(self.index, self.index + self.batch_size)
            ])

            # resize(image=,
            # output_shape=output_shape,
            #         mode='constant',
            #         preserve_range=True,
            #         anti_aliasing=True)

            # new_segmentation = np.zeros(
            #     shape=(self.batch_size, output_shape[0], output_shape[1],
            #            len(self.seg_dic)))
            # for i in range(self.batch_size):
            #     for j in range(output_shape[0]):
            #         for k in range(output_shape[1]):
            #             new_segmentation[i, j, k, 
            #             self.seg_dic[ 
            #                 tuple(segmentation[i, j, k]) ][0]] = 1
            # segmentation = new_segmentation
            
            if self.data_type is 'float32':
                segmentation = img_as_float32(segmentation)
            else:
                segmentation = img_as_float64(segmentation)

        # if self.label_type == 'depth':
        #     labels = np.array(
        #         np.array([
        #             resize(
        #                 image=imread(self.dataset.iloc[i, 1]),
        #                 output_shape=(480, 640))
        #             for i in range(self.index, self.index + self.batch_size)
        #         ]),
        #         dtype=np.int32)
        #     labels = (labels[:, :, :, 0] + labels[:, :, :, 1] * 256 +
        #               labels[:, :, :, 2] * 256 * 256) / ((256 * 256 * 256) - 1)

        # elif self.label_type == 'segmentation':
        #     labels = np.array(
        #         np.array([
        #             resize(
        #                 image=imread(self.dataset.iloc[i, 2])[:, :, 0],
        #                 output_shape=(480, 640))
        #             for i in range(self.index, self.index + self.batch_size)
        #         ]))

        #     new_segmentation = np.ndarray(shape=(self.batch_size, 480, 640, 22))
        #     for i in range(self.batch_size):
        #         for j in range(480):
        #             for k in range(640):
        #                 if labels[i, j, k] < 22:
        #                     new_segmentation[i, j, k, int(labels[i, j, k])] = 1
        #     labels = new_segmentation
        #     labels = np.array(labels, dtype=np.float32)
        # else:
        #     raise ValueError('invalid label type')

        # return features, labels
        return features, segmentation


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
        features = np.array(features / 255., dtype=np.float16)

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
            new_segmentation = np.ndarray(
                shape=(self.batch_size, 480, 640, 21))

            for i in range(self.batch_size):
                for j in range(480):
                    for k in range(640):
                        if labels[i, j, k] < 21:
                            new_segmentation[i, j, k, int(labels[i, j, k])] = 1
            labels = new_segmentation
            labels = np.array(labels, dtype=np.float16)
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
                resize(
                    image=imread(self.dataset.iloc[i, 0])[:, :, :3],
                    output_shape=(480, 640))
                for i in range(self.index, self.index + self.batch_size)
            ]))
        # features = np.array(features / 255., dtype=np.float32)

        if self.label_type == 'depth':
            labels = np.array(
                np.array([
                    resize(
                        image=imread(self.dataset.iloc[i, 1]),
                        output_shape=(480, 640))
                    for i in range(self.index, self.index + self.batch_size)
                ]),
                dtype=np.int32)
            labels = (labels[:, :, :, 0] + labels[:, :, :, 1] * 256 +
                      labels[:, :, :, 2] * 256 * 256) / ((256 * 256 * 256) - 1)

        elif self.label_type == 'segmentation':
            labels = np.array([
                    resize(
                        image=imread(self.dataset.iloc[i, 2])[:, :, 0],
                        output_shape=(480, 640))
                    for i in range(self.index, self.index + self.batch_size)
                ])
            labels = img_as_ubyte(labels)

            new_segmentation = np.ndarray(
                shape=(self.batch_size, 480, 640, 22))
            for i in range(self.batch_size):
                for j in range(480):
                    for k in range(640):
                        if labels[i, j, k] < 22:
                            new_segmentation[i, j, k, int(labels[i, j, k])] = 1
            labels = new_segmentation

        elif self.label_type == 'sparse_segmentation':
            labels = np.array([
                    resize(
                        image=imread(self.dataset.iloc[i, 2])[:, :, 0],
                        output_shape=(480, 640, 1))
                    for i in range(self.index, self.index + self.batch_size)
                ])
            labels = img_as_ubyte(labels)

            # new_segmentation = np.ndarray(
            #     shape=(self.batch_size, 480, 640, 22))
            # for i in range(self.batch_size):
            #     for j in range(480):
            #         for k in range(640):
            #             if labels[i, j, k] < 22:
            #                 new_segmentation[i, j, k, int(labels[i, j, k])] = 1
            # labels = new_segmentation
            # labels = np.array(labels, dtype=np.float32)
            # labels = img_as_ubyte(labels)
        else:
            raise ValueError('invalid label type')

        return features, labels


class VIPER(DatasetGenerator):
    """Iterator for looping over elements of a VIPER(PlayForBenchmark) dataset ."""

    def init(self):
        self.seg_dic = segmentation_dics.VIPER

    def data_frame_creator(self):
        """Pandas dataFrame for addresses of images and corresponding labels"""
        img_dir_list = [
            self.dataset_dir + '/train' + '/img' + '/' + seq_dir + '/' +
            img_name
            for seq_dir in os.listdir(self.dataset_dir + '/train/' + '/img/')
            for img_name in os.listdir(self.dataset_dir + '/train/' + '/img/' +
                                       seq_dir)
        ]

        cls_dir_list = [
            self.dataset_dir + '/train' + '/cls' + '/' + seq_dir + '/' +
            img_name
            for seq_dir in os.listdir(self.dataset_dir + '/train/' + '/cls/')
            for img_name in os.listdir(self.dataset_dir + '/train/' + '/cls/' +
                                       seq_dir)
        ]

        inst_dir_list = [
            self.dataset_dir + '/train' + '/inst' + '/' + seq_dir + '/' +
            img_name
            for seq_dir in os.listdir(self.dataset_dir + '/train/' + '/inst/')
            for img_name in os.listdir(self.dataset_dir + '/train/' +
                                       '/inst/' + seq_dir)
        ]

        dataset = {
            'image': img_dir_list,
            'segmentation': cls_dir_list,
            'instance': inst_dir_list
        }

        if self.shuffle:
            return pd.DataFrame(dataset).sample(frac=1)
        return pd.DataFrame(dataset)
