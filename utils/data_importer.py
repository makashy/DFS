"""Contains dataset importers for NYU Depth Dataset V2 and SYNTHIA-SF"""

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pandas as pd
import tables
from numba import jit
from skimage import img_as_float32, img_as_ubyte
from skimage.io import imread

# from skimage.transform import resize
from cv2 import resize

COMMON_LABEL_IDS = [
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
            class_ids=COMMON_LABEL_IDS,
            dataset_name='SYNTHIA_SF'):
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
           class_ids=COMMON_LABEL_IDS,
           dataset_name='SYNTHIA_SF'):
    """ Creates a sparse label for a group of segmentation classes from the given raw label """
    label = np.zeros(shape=(raw_label.shape[0], raw_label.shape[1], 1),
                     dtype=np.uint8)
    for i, class_id in enumerate(class_ids):
        class_color = label_table.loc[class_id, dataset_name]
        label[:, :, 0] = label[:, :, 0] + label_slicer(raw_label,
                                                       class_color) * i
    return label


class DatasetGenerator:
    """Abstract iterator for looping over elements of a dataset .

    Arguments:
        usage_range: usage range of the dataset
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
                 table_address,
                 usage='train',
                 usage_range=(0, 1),
                 batch_size=1,
                 repeater=True,
                 shuffle=True,
                 output_shape=None,
                 data_type='float32',
                 label_type=('segmentation'),
                 dataset_name=None):
        self.usage = usage
        self.usage_range = usage_range
        self.batch_size = batch_size
        self.repeater = repeater
        self.shuffle = shuffle
        self.output_shape = output_shape
        self.data_type = data_type
        self.label_type = label_type
        self.dataset_name = dataset_name
        self.dataset = self.data_frame_creator()
        self.start_index = np.int32(np.floor(self.usage_range[0] * (self.dataset.shape[0] - 1)))
        self.end_index = np.int32(np.floor(self.usage_range[1] * (self.dataset.shape[0] - 1)))
        self.size = self.end_index - self.start_index
        self.index = self.start_index
        self.label_table = self.load_label_table(table_address)

    def data_frame_creator(self):
        """Pandas dataFrame for addresses of images and corresponding labels"""

        return pd.DataFrame()

    def load_label_table(self, table_address):
        """ Creates a pandas data frame (from a CSV file) having information
        about segmentation class colors """
        label_table = pd.read_csv(table_address, index_col=0)
        for dataset_name in label_table.columns[3:]:
            color_list = label_table.loc[:, dataset_name]
            for i, color in enumerate(color_list):
                if color != 'None':
                    color_list[i] = np.fromstring(color_list[i][1:-1],
                                                  dtype=np.uint8,
                                                  sep=',')
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
        features = imread(self.dataset.RGB[0], plugin='matplotlib')[:, :, :3]

        if self.output_shape is None:
            output_shape = features.shape[:2]
        else:
            output_shape = self.output_shape

        # 1) Resize image to match a certain size.
        # 2) Also the input image is converted (from 8-bit integer)
        # to 64-bit floating point(->preserve_range=False).
        # 3) [:, :, :3] -> to remove 4th channel in png

        features = np.array([
            resize(src=imread(self.dataset.RGB[i], plugin='matplotlib')[:, :, :3],
                   dsize=(output_shape[1], output_shape[0]))
            for i in range(self.index - self.batch_size, self.index)
        ])

        if self.data_type is 'float32':
            features = img_as_float32(features)

        # loading labels(segmentation)
        if self.label_type == 'segmentation':
            segmentation = np.array([
                one_hot(self.label_table,
                        img_as_ubyte(
                            resize(src=imread(
                                self.dataset.SEGMENTATION[i], plugin='matplotlib')[:, :, :3],
                                   dsize=(output_shape[1], output_shape[0]))),
                        dataset_name=self.dataset_name)
                for i in range(self.index - self.batch_size, self.index)
            ])

        if self.label_type == 'sparse_segmentation':
            segmentation = np.array([
                sparse(self.label_table,
                       img_as_ubyte(
                           resize(src=imread(
                               self.dataset.SEGMENTATION[i], plugin='matplotlib')[:, :, :3],
                                  dsize=(output_shape[1], output_shape[0]))),
                       dataset_name=self.dataset_name)
                for i in range(self.index - self.batch_size, self.index)
            ])

        # if self.label_type == 'depth':
        #     labels = np.array(
        #         np.array([
        #             resize(
        #                 image=imread(self.dataset.iloc[i, 1], plugin='matplotlib'),
        #                 output_shape=(480, 640))
        #             for i in range(self.index, self.index + self.batch_size)
        #         ]),
        #         dtype=np.int32)
        #     labels = (labels[:, :, :, 0] + labels[:, :, :, 1] * 256 +
        #               labels[:, :, :, 2] * 256 * 256) / ((256 * 256 * 256) - 1)
        return features, segmentation


class SynthiaSf(DatasetGenerator):
    """Iterator for looping over elements of SYNTHIA-SF backwards."""

    def __init__(self, dataset_dir, **kwargs):

        self.dataset_dir = dataset_dir
        self.max_distance = 1000
        super().__init__(dataset_name='SYNTHIA_SF', **kwargs)

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
            self.dataset_dir + sequence_f + depth_f for depth_f in depth_folder
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
            return pd.DataFrame(dataset).sample(
                frac=1, random_state=123).reset_index(drop=True)

        return pd.DataFrame(dataset)


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


class VIPER(DatasetGenerator):
    """Iterator for looping over elements of a VIPER(PlayForBenchmark) dataset ."""

    def __init__(self, dataset_dir, **kwargs):

        self.dataset_dir = dataset_dir
        super().__init__(dataset_name='VIPER', **kwargs)

    def data_frame_creator(self):
        """Pandas dataFrame for addresses of images and corresponding labels"""

        if self.usage == 'train':
            main_folder = '/train'

        elif self.usage == 'validation':
            main_folder = '/val'

        else:
            main_folder = '/test'

        img_dir_list = [
            self.dataset_dir + main_folder + '/img' + '/' + seq_dir + '/' +
            img_name
            for seq_dir in os.listdir(self.dataset_dir + main_folder + '/img/')
            for img_name in os.listdir(self.dataset_dir + main_folder +
                                       '/img/' + seq_dir)
        ]

        cls_dir_list = [
            self.dataset_dir + main_folder + '/cls' + '/' + seq_dir + '/' +
            img_name
            for seq_dir in os.listdir(self.dataset_dir + main_folder + '/cls/')
            for img_name in os.listdir(self.dataset_dir + main_folder +
                                       '/cls/' + seq_dir)
        ]

        inst_dir_list = [
            self.dataset_dir + main_folder + '/inst' + '/' + seq_dir + '/' +
            img_name for seq_dir in os.listdir(self.dataset_dir + main_folder +
                                               '/inst/')
            for img_name in os.listdir(self.dataset_dir + main_folder +
                                       '/inst/' + seq_dir)
        ]

        dataset = {
            'RGB': img_dir_list,
            'SEGMENTATION': cls_dir_list,
            'INSTANCE': inst_dir_list
        }

        if self.shuffle:
            return pd.DataFrame(dataset).sample(
                frac=1, random_state=123).reset_index(drop=True)
        return pd.DataFrame(dataset)


class MAPILLARY(DatasetGenerator):
    """Iterator for looping over elements of a MAPILLARY dataset ."""

    def __init__(self, dataset_dir, **kwargs):

        self.dataset_dir = dataset_dir
        super().__init__(dataset_name='MAPILLARY', **kwargs)

    def data_frame_creator(self):
        """Pandas dataFrame for addresses of images and corresponding labels"""
        if self.usage == 'train':
            main_folder = '/training'

        elif self.usage == 'validation':
            main_folder = '/validation'

        else:
            main_folder = '/testing'

        img_dir_list = [
            self.dataset_dir + main_folder + '/images' + '/' + img_name
            for img_name in os.listdir(self.dataset_dir + main_folder +
                                       '/images/')
        ]

        cls_dir_list = [
            self.dataset_dir + main_folder + '/labels' + '/' + img_name
            for img_name in os.listdir(self.dataset_dir + main_folder +
                                       '/labels/')
        ]

        inst_dir_list = [
            self.dataset_dir + main_folder + '/instances' + '/' + img_name
            for img_name in os.listdir(self.dataset_dir + main_folder +
                                       '/instances/')
        ]

        dataset = {
            'RGB': img_dir_list,
            'SEGMENTATION': cls_dir_list,
            'INSTANCE': inst_dir_list
        }

        if self.shuffle:
            return pd.DataFrame(dataset).sample(
                frac=1, random_state=123).reset_index(drop=True)
        return pd.DataFrame(dataset)


class CITYSCAPES(DatasetGenerator):
    """Iterator for looping over elements of a CITYSCAPES dataset ."""

    def __init__(self, dataset_dir, **kwargs):

        self.dataset_dir = dataset_dir
        super().__init__(dataset_name='CITYSCAPES', **kwargs)

    def data_frame_creator(self):
        """Pandas dataFrame for addresses of images and corresponding labels"""

        if self.usage == 'train':
            main_folder = '/train'

        elif self.usage == 'validation':
            main_folder = '/val'

        else:
            main_folder = '/test'

        img_dir_list = [
            self.dataset_dir + '/leftImg8bit_trainvaltest/leftImg8bit' +
            main_folder + '/' + seq_dir + '/' + img_name
            for seq_dir in os.listdir(self.dataset_dir +
                                      '/leftImg8bit_trainvaltest/leftImg8bit' +
                                      main_folder)
            for img_name in os.listdir(
                self.dataset_dir + '/leftImg8bit_trainvaltest/leftImg8bit' +
                main_folder + '/' + seq_dir)
        ]

        def get_label(address, label_type):
            raw_list = os.listdir(address)
            final_list = list()
            for _, element in enumerate(raw_list):
                if label_type in element:
                    final_list.append(element)
            return final_list

        cls_dir_list = [
            self.dataset_dir + '/gtFine_trainvaltest/gtFine' + main_folder +
            '/' + seq_dir + '/' + img_name
            for seq_dir in os.listdir(self.dataset_dir +
                                      '/gtFine_trainvaltest/gtFine' +
                                      main_folder)
            for img_name in get_label(
                self.dataset_dir + '/gtFine_trainvaltest/gtFine' +
                main_folder + '/' + seq_dir, "color")
        ]

        inst_dir_list = [
            self.dataset_dir + '/gtFine_trainvaltest/gtFine' + main_folder +
            '/' + seq_dir + '/' + img_name
            for seq_dir in os.listdir(self.dataset_dir +
                                      '/gtFine_trainvaltest/gtFine' +
                                      main_folder)
            for img_name in get_label(
                self.dataset_dir + '/gtFine_trainvaltest/gtFine' +
                main_folder + '/' + seq_dir, "instanceIds")
        ]

        dataset = {
            'RGB': img_dir_list,
            'SEGMENTATION': cls_dir_list,
            'INSTANCE': inst_dir_list
        }

        if self.shuffle:
            return pd.DataFrame(dataset).sample(
                frac=1, random_state=123).reset_index(drop=True)
        return pd.DataFrame(dataset)
