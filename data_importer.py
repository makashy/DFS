"""Contains dataset importers for NYU Depth Dataset V2 and SYNTHIA-SF"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd
import tables
from skimage import img_as_float32
# from skimage import img_as_float64
from skimage.io import imread
from skimage.transform import resize
from skimage import img_as_ubyte
# from skimage import img_as_uint
from numba import jit

# import segmentation_dics

COMMON_LABE_IDS = [
    3, 13, 15, 17, 19, 20, 27, 29, 30, 45, 48, 50, 52, 54, 55, 57, 58, 61, 65
]

RGB = 0
SEGMENTATION = 1
INSTANCE = 2
DEPTH = 3

TRAIN = 0
VALIDATION = 1
TEST = 2

@jit(nopython=True)
def label_slicer(raw_label, class_color):
    """ Creates a channel for a specific segmentation class from the given raw label """
    return (raw_label[:, :, 0] == class_color[0])* \
           (raw_label[:, :, 1] == class_color[1])* \
           (raw_label[:, :, 2] == class_color[2])


def one_hot(label_table,
            raw_label,
            class_ids=COMMON_LABE_IDS,
            dataset_name='SYNTHIA'):
    """ Creates a one-hot label for a group of segmentation classes from the given raw label """
    one_hot_label = np.ndarray(shape=(raw_label.shape[0], raw_label.shape[1],
                                      len(class_ids)),
                               dtype=np.uint8)
    for i, class_id in enumerate(class_ids):
        class_color = label_table.loc[class_id, dataset_name]
        one_hot_label[:, :, i] = label_slicer(raw_label, class_color)
    return one_hot_label

def sparse(label_table,
           raw_label,
           class_ids=COMMON_LABE_IDS,
           dataset_name='SYNTHIA'):
    """ Creates a one-hot label for a group of segmentation classes from the given raw label """
    label = np.zeros(shape=(raw_label.shape[0], raw_label.shape[1], 1), dtype=np.uint8)
    for i, class_id in enumerate(class_ids):
        class_color = label_table.loc[class_id, dataset_name]
        label[:, :, 0] = label[:, :, 0] + label_slicer(raw_label, class_color)*i
    return label


class DatasetGenerator:
    """Abstract iterator for looping over elements of a dataset .

    Arguments:
        ratio: ratio of the train-set size to the validation-set size and test-set size
            The first number is for the train-set, the second is for validation-set and what
            is remained is for test-set.(the sum of two numbers should equal to one or less)
        batch_size: Integer batch size.
        repeater: If true, the dataset generator starts generating samples from the beginning when
            it reaches the end of the dataset.
        shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of the
            training.
        output_shape: size of generated images and labels.
        data_type: data type of features.
        label_type: Types of labels to be returned.
    """

    def __init__(self,
                 usage='train',
                 ratio=(1, 0),
                 batch_size=1,
                 repeater=False,
                 shuffle=True,
                 output_shape=None,
                 data_type='float64',
                 label_type=('segmentation', 'instance', 'depth'),
                 **kwargs):
        self.ratio = kwargs[
            'ratio'] if 'ratio' in kwargs else ratio
        self.batch_size = kwargs[
            'batch_size'] if 'batch_size' in kwargs else batch_size
        self.repeater = kwargs['repeater'] if 'repeater' in kwargs else repeater
        self.shuffle = kwargs['shuffle'] if 'shuffle' in kwargs else shuffle
        self.output_shape = kwargs[
            'output_shape'] if 'output_shape' in kwargs else output_shape
        self.data_type = kwargs[
            'data_type'] if 'data_type' in kwargs else data_type
        self.label_type = kwargs[
            'label_type'] if 'label_type' in kwargs else label_type
        self.dataset = self.data_frame_creator()
        self.size = self.dataset.shape[0] - 1
        self.start_index = 0
        self.end_index = np.int32(np.floor(self.ratio[TRAIN] * self.size))
        self.dataset_usage(usage)
        self.index = self.start_index
        self.label_table = self.load_label_table(kwargs['table_address'])

    def data_frame_creator(self):
        """Pandas dataFrame for addresses of images and corresponding labels"""

        return pd.DataFrame()

    def dataset_usage(self, usage):
        """ Determines the current usage of the dataset:
            - 'train'
            - 'validation'
            - 'test'
        """
        if usage is 'train':
            self.start_index = 0
            self.end_index = np.int32(np.floor(self.ratio[TRAIN] * self.size))
        elif usage is 'validation':
            self.start_index = np.int32(np.floor(self.ratio[TRAIN] * self.size))
            self.end_index = np.int32(np.floor((self.ratio[TRAIN] + self.ratio[VALIDATION])* self.size))
        elif usage is 'test':
            self.start_index = np.int32(np.floor((self.ratio[TRAIN] + self.ratio[VALIDATION])* self.size))
            self.end_index = self.size
        else:
            print('Invalid input for usage variable')
            raise NameError('InvalidInput')

    def load_label_table(self, table_address):
        """ Creates a pandas data frame (from a CSV file) having information
        about segmentation class colors """
        label_table = pd.read_csv(table_address, index_col=0)
        for dataset_name in label_table.columns[3:]:
            color_list = label_table.loc[:, dataset_name]
            for i, color in enumerate(color_list):
                if color != 'None':
                    color_list[i] = np.fromstring(color_list[i][1:-1], dtype=np.uint8, sep=',')
        return label_table

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        """Retrieve the next pairs from the dataset"""

        if self.index + self.batch_size > self.end_index:
            if not self.repeater:
                raise StopIteration
            else:
                self.index = self.start_index
        self.index = self.index + self.batch_size

        # loading features(images)
        features = imread(self.dataset.RGB[0])[:, :, :3]

        if self.output_shape is None:
            output_shape = features.shape[:2]
        else:
            output_shape = self.output_shape

        # 1) Resize image to match a certain size.
        # 2) Also the input image is converted (from 8-bit integer)
        # to 64-bit floating point(->preserve_range=False).
        # 3) [:, :, :3] -> to remove 4th channel in png

        features = np.array([
            resize(image=imread(self.dataset.RGB[i])[:, :, :3],
                   output_shape=output_shape,
                   mode='constant',
                   preserve_range=False,
                   anti_aliasing=True)
            for i in range(self.index - self.batch_size, self.index)
        ])

        if self.data_type is 'float32':
            features = img_as_float32(features)

        # loading labels(segmentation)
        if self.label_type == 'segmentation':
            segmentation = np.array([
                one_hot(
                    self.label_table,
                    img_as_ubyte(
                        resize(image=imread(
                            self.dataset.SEGMENTATION[i])[:, :, :3],
                               output_shape=self.output_shape)))
                for i in range(self.index - self.batch_size, self.index)
            ])

        if self.label_type == 'sparse_segmentation':
            segmentation = np.array([
                sparse(
                    self.label_table,
                    img_as_ubyte(
                        resize(image=imread(
                            self.dataset.SEGMENTATION[i])[:, :, :3],
                               output_shape=self.output_shape)))
                for i in range(self.index - self.batch_size, self.index)
            ])

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

        return features, segmentation


class NewSynthiaSf(DatasetGenerator):
    """Iterator for looping over elements of SYNTHIA-SF backwards."""

    def __init__(self, synthia_sf_dir, **kwargs):

        self.dataset_dir = synthia_sf_dir
        self.max_distance = 1000
        super().__init__(**kwargs)

    def data_frame_creator(self):
        """ pandas dataFrame for addresses of rgb, depth and segmentation"""
        sequence_folder = [
            '/SEQ1', '/SEQ2', '/SEQ3', '/SEQ4', '/SEQ5', '/SEQ6'
        ]
        rgb_folder = ['/RGBLeft/', '/RGBRight/']
        depth_folder = ['/DepthLeft/', '/DepthRight/']
        segmentation_folder = ['/GTLeftDebug/', '/GTrightDebug/']
        rgb_dir = [
            self.dataset_dir + sequence_f + rgb_f for rgb_f in rgb_folder
            for sequence_f in sequence_folder
        ]
        rgb_data = [
            rgb_d + rgb for rgb_d in rgb_dir for rgb in os.listdir(rgb_d)
        ]

        depth_dir = [
            self.dataset_dir + sequence_f + depth_f
            for depth_f in depth_folder
            for sequence_f in sequence_folder
        ]
        depth_data = [
            depth_d + depth for depth_d in depth_dir
            for depth in os.listdir(depth_d)
        ]

        segmentation_dir = [
            self.dataset_dir + sequence_f + segmentation_f
            for segmentation_f in segmentation_folder
            for sequence_f in sequence_folder
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
            return pd.DataFrame(dataset).sample(frac=1, random_state=123)

        return pd.DataFrame(dataset)

    # def next(self):
    #     """Retrieve the next pairs from the dataset"""

    #     if self.index + self.batch_size > self.end_index:
    #         if not self.repeater:
    #             raise StopIteration
    #         else:
    #             self.index = self.start_index
    #     self.index = self.index + self.batch_size

    #     features = np.array([
    #         resize(image=imread(self.dataset.iloc[i, 0])[:, :, :3],
    #                output_shape=self.output_shape)
    #         for i in range(self.index - self.batch_size, self.index)
    #     ])
    #     features = np.array(features, dtype=np.float32)

    #     if self.label_type == 'depth':
    #         labels = np.array([
    #             resize(image=imread(self.dataset.iloc[i, 1]),
    #                    output_shape=self.output_shape)
    #             for i in range(self.index, self.index + self.batch_size)
    #         ])
    #         labels = img_as_ubyte(labels)
    #         labels = np.array(labels, dtype=np.float)
    #         labels = (labels[:, :, :, 0] + labels[:, :, :, 1] * 256 +
    #                   labels[:, :, :, 2] * 256 * 256) / (
    #                       (256 * 256 * 256) - 1) * self.max_distance

    #     elif self.label_type == 'segmentation':
    #         labels = np.array([
    #             resize(image=imread(self.dataset.iloc[i, 2])[:, :, 0],
    #                    output_shape=self.output_shape)
    #             for i in range(self.index - self.batch_size, self.index)
    #         ])
    #         labels = img_as_ubyte(labels)

    #         new_segmentation = np.ndarray(shape=(self.batch_size, 480, 640,
    #                                              22))

    #         for i in range(self.batch_size):
    #             for j in range(480):
    #                 for k in range(640):
    #                     if labels[i, j, k] < 22:
    #                         new_segmentation[i, j, k, int(labels[i, j, k])] = 1
    #         labels = new_segmentation

    #     elif self.label_type == 'sparse_segmentation':
    #         labels = np.array([
    #             resize(image=imread(self.dataset.iloc[i, 2])[:, :, 0],
    #                    output_shape=self.output_shape + (1, ))
    #             for i in range(self.index - self.batch_size, self.index)
    #         ])
    #         labels = img_as_ubyte(labels)
    #     else:
    #         raise ValueError('invalid label type')

    #     return features, labels


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

        features = np.array(self.file.root.images[self.index:self.index +
                                                  self.batch_size])
        features = np.transpose(features, [0, 3, 2, 1])
        features = np.array(features / 255., dtype=np.float16)

        if self.label_type == 'depth':
            label_set = self.file.root.depths
            labels = np.array(label_set[self.index:self.index +
                                        self.batch_size])
            labels = np.transpose(labels, [0, 2, 1])
            labels = np.reshape(
                labels, (labels.shape[0], labels.shape[1], labels.shape[2], 1))
        elif self.label_type == 'segmentation':
            label_set = self.file.root.labels
            labels = np.array(label_set[self.index:self.index +
                                        self.batch_size])
            labels = np.transpose(labels, [0, 2, 1])
            labels = np.reshape(
                labels, (labels.shape[0], labels.shape[1], labels.shape[2], 1))
            new_segmentation = np.ndarray(shape=(self.batch_size, 480, 640,
                                                 21))

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


# class VIPER(DatasetGenerator):
#     """Iterator for looping over elements of a VIPER(PlayForBenchmark) dataset ."""

#     def init(self):
#         self.seg_dic = segmentation_dics.VIPER

#     def data_frame_creator(self):
#         """Pandas dataFrame for addresses of images and corresponding labels"""
#         img_dir_list = [
#             self.dataset_dir + '/train' + '/img' + '/' + seq_dir + '/' +
#             img_name
#             for seq_dir in os.listdir(self.dataset_dir + '/train/' + '/img/')
#             for img_name in os.listdir(self.dataset_dir + '/train/' + '/img/' +
#                                        seq_dir)
#         ]

#         cls_dir_list = [
#             self.dataset_dir + '/train' + '/cls' + '/' + seq_dir + '/' +
#             img_name
#             for seq_dir in os.listdir(self.dataset_dir + '/train/' + '/cls/')
#             for img_name in os.listdir(self.dataset_dir + '/train/' + '/cls/' +
#                                        seq_dir)
#         ]

#         inst_dir_list = [
#             self.dataset_dir + '/train' + '/inst' + '/' + seq_dir + '/' +
#             img_name
#             for seq_dir in os.listdir(self.dataset_dir + '/train/' + '/inst/')
#             for img_name in os.listdir(self.dataset_dir + '/train/' +
#                                        '/inst/' + seq_dir)
#         ]

#         dataset = {
#             'image': img_dir_list,
#             'segmentation': cls_dir_list,
#             'instance': inst_dir_list
#         }

#         if self.shuffle:
#             return pd.DataFrame(dataset).sample(frac=1)
#         return pd.DataFrame(dataset)
