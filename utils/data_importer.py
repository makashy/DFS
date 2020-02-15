"""Contains dataset importers for NYU Depth Dataset V2 and SYNTHIA-SF"""

from __future__ import absolute_import, division, print_function

import os

import cv2
import numpy as np
import pandas as pd
import tables
from lyft_dataset_sdk.lyftdataset import LyftDataset, LyftDatasetExplorer
from numba import jit
from skimage import img_as_float32, img_as_ubyte
from skimage.io import imread
from tensorflow.python.keras.utils.data_utils import \
    Sequence  # pylint: disable=import-error,no-name-in-module

COMMON_LABEL_IDS = [
    3, 13, 15, 17, 19, 20, 27, 29, 30, 45, 48, 50, 52, 54, 55, 57, 58, 61, 65
]

@jit(nopython=True)
def label_slicer(raw_label, class_color):
    """ Creates a channel for a specific segmentation class from the given raw label """
    return (raw_label[:, :, 0] == class_color[0])* \
           (raw_label[:, :, 1] == class_color[1])* \
           (raw_label[:, :, 2] == class_color[2])


def one_hot(label_table,
            raw_label,
            class_ids=COMMON_LABEL_IDS.copy(),
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
           class_ids=COMMON_LABEL_IDS.copy(),
           dataset_name='SYNTHIA_SF'):
    """ Creates a sparse label for a group of segmentation classes from the given raw label """
    label = np.zeros(shape=(raw_label.shape[0], raw_label.shape[1], 1),
                     dtype=np.uint8)
    for i, class_id in enumerate(class_ids):
        class_color = label_table.loc[class_id, dataset_name]
        label[:, :, 0] = label[:, :, 0] + label_slicer(raw_label,
                                                       class_color) * i
    return label


def resize_and_split(input_array, output_shape, max_split):
    """ Splits the input_array into several windows with shape of output_shape.
        max_split determines the maximum number of splits in each direction.
        If the input_array is too large to be covered by max_split windows, then
        it will be resized before splitting to be consistant with max_split.
    """
    o_height = output_shape[0]
    o_width = output_shape[1]
    i_height = input_array.shape[0]
    i_width = input_array.shape[1]

    h_ratio = i_height / o_height
    w_ratio = i_width / o_width
    max_ratio = max(h_ratio, w_ratio)
    resize_ratio = max_split / max_ratio
    dims = (int(i_height * resize_ratio), int(i_width * resize_ratio))
    # if (i_height*resize_ratio + i_width*resize_ratio) % 1 > 0:
    #     print("Cropping occurred in resize!")

    row_num = int(np.ceil(dims[0] / o_height))
    column_num = int(np.ceil(dims[1] / o_width))
    column_step = int(np.floor((dims[0] - o_height) / row_num))
    row_step = int(np.floor((dims[1] - o_width) / column_num))

    resized_img = cv2.resize(input_array, (dims[1], dims[0]),
                             interpolation=cv2.INTER_NEAREST)
    if len(resized_img.shape) == 2:
        resized_img = np.expand_dims(resized_img, -1)

    result = []
    for row in range(row_num):
        for column in range(column_num):
            result.append(resized_img[column_step * row:o_height +
                                      column_step * row, row_step *
                                      column:o_width + row_step * column])
    return result


def resize(array, resize_info, interpolation=cv2.INTER_NEAREST):
    """ Resizes the array according to resize_info.
        resize_info can be a tuple (dual) or a float number
        to determine output shape or resizing ratio respectively.
    """
    if isinstance(resize_info, tuple):
        result = cv2.resize(src=array,
                            dsize=(resize_info[1], resize_info[0]),
                            interpolation=interpolation)
    else:
        result = cv2.resize(src=array,
                            dsize=(int(array.shape[1] * resize_info),
                                   int(array.shape[0] * resize_info)),
                            interpolation=interpolation)
    if len(result.shape) == 2:
        result = np.expand_dims(result, -1)
    return result


class DatasetGenerator(Sequence):
    """Abstract iterator for looping over elements of a dataset .

    Arguments:
        usage_range: usage range of the dataset
        batch_size: Integer batch size.
        repeater: If true, the dataset generator starts generating samples from the beginning when
            it reaches the end of the dataset.
        shuffle: Boolean. Whether to shuffle the order of the batches at the beginning of the
            training.
        shape: size of generated images and labels.
        data_type: data type of features.
        label_type: Types of labels to be returned.
    """
    def __init__(self,
                 table_address="./utils/labels.csv",
                 dataset_name=None,
                 feature_types=None,
                 label_types=None,
                 class_ids='all',
                 output_shape=(512, 512),
                 float_type='float32',
                 focal_length=None,
                 split=True,
                 max_split=3,
                 usage='train',
                 usage_range=(0, 1),
                 batch_size=1,
                 repeater=True,
                 shuffle=True):
        self.label_table = self.load_label_table(table_address)
        self.dataset_name = dataset_name
        self.feature_types = feature_types
        self.label_types = label_types
        self.class_ids = self.available_classes(
        ) if class_ids is 'all' else class_ids
        self.output_shape = output_shape
        self.float_type = float_type
        self.focal_length = focal_length
        self.split = split
        self.max_split = max_split

        self.batch_size = batch_size
        self.repeater = repeater
        self.dataset = self.data_frame_creator(usage, shuffle)
        #####################################################################
        self.data_list = []
        for cat in self.available_categories:
            if cat in feature_types + label_types:
                self.data_list.append(cat)

        for cat in self.feature_types + self.label_types:
            try:
                if not cat in self.available_categories:
                    raise NameError('InvalidCategory')
            except NameError:
                print("Oops! This data category (" + cat + ") is not available in "
                      + "this dataset or this data generator does not support it :(")

        self.data_buffer = {key: [] for key in self.data_list}
        self.data_history = {key: None for key in self.data_list}
        self.indexes = self.determine_indexes(usage_range)
        #####################################################################

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return np.int(
            np.ceil((self.indexes['end'] - self.indexes['start']) /
                    self.batch_size))

    def __getitem__(self, idx):
        return self.next()

    def data_frame_creator(self, usage, shuffle):  #pylint: disable=unused-argument
        """Pandas dataFrame for addresses of images and corresponding labels"""
        return pd.DataFrame()

    def load_data(self, data_type, index, resize_info=1):
        """Loads requested data from dataset and does some preprocessing:
            1. resizes arrayes
            2. generates coplex data from raw data (like semantic depth from semantic segmentation)
        """
        if data_type is "focal_length":
            focal_length = self.dataset.FOCAL_LENGTH[index]
            return focal_length

        if data_type is "image":
            image = imread(self.dataset.RGB[index],
                           plugin='matplotlib')[:, :, :3]
            image = resize(image, resize_info)
            if self.float_type is 'float32':
                image = img_as_float32(image)
            return image

        if data_type is "segmentation" or data_type is "semantic_depth":
            array = img_as_ubyte(
                imread(self.dataset.SEGMENTATION[index],
                       plugin='matplotlib')[:, :, :3])
            array = resize(array, resize_info)
            segmentation = one_hot(self.label_table,
                                   array,
                                   class_ids=self.class_ids,
                                   dataset_name=self.dataset_name)
            if len(segmentation.shape) == 2:
                segmentation = np.expand_dims(segmentation, -1)
            if self.float_type is 'float32':
                segmentation = np.array(segmentation, dtype=np.float32)
            self.data_history["segmentation"] = segmentation
            if data_type is "segmentation":
                return segmentation

        if data_type is "depth" or data_type is "semantic_depth":
            depth = imread(self.dataset.DEPTH[index], plugin='pil')
            depth = resize(depth, resize_info)
            depth = np.array(
                (depth[:, :, 0] + depth[:, :, 1] * 256.0 +
                 depth[:, :, 2] * 256 * 256.0) / ((256 * 256 * 256) - 1),
                dtype=np.float32) * 1000
            if self.float_type is 'float32':
                depth = img_as_float32(depth)
            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, -1)
            self.data_history["depth"] = depth
            if data_type is "depth":
                return depth

        if data_type is "sparse_segmentation":
            array = img_as_ubyte(
                imread(self.dataset.SEGMENTATION[index],
                       plugin='matplotlib')[:, :, :3])
            array = resize(array, resize_info)
            return sparse(self.label_table,
                          array,
                          class_ids=self.class_ids,
                          dataset_name=self.dataset_name)

        if data_type is "semantic_depth":
            semantic_depth_array = self.data_history[
                "segmentation"] * self.data_history["depth"]
            ######
            # semantic_depth_array = np.rollaxis(semantic_depth_array, -1, 0)
            # semantic_depth = []
            # for array in semantic_depth_array:
            #     semantic_depth.append(np.expand_dims(array, -1))
            #####
            semantic_depth = semantic_depth_array
            return semantic_depth

    def determine_indexes(self, usage_range):
        """ Calculates and returns start and end indexes of the generator.
            Also initialize the current index.
        """
        if self.split is True:
            image = self.load_data('image', 0)
            split_coefficient = len(
                resize_and_split(image, self.output_shape, self.max_split))
        else:
            split_coefficient = 1
        start_index = np.int32(
            np.floor(usage_range[0] * (self.dataset.shape[0] - 1) *
                     split_coefficient))
        end_index = np.int32(
            np.floor(usage_range[1] * (self.dataset.shape[0] - 1) *
                     split_coefficient))
        return {
            'start': start_index,
            'end': end_index,
            'current': start_index,
            'split_coefficient': split_coefficient
        }

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

    def available_classes(self):
        """Returns indexes of available classes in the dataset
        """
        return self.label_table[self.dataset_name].index[(
            self.label_table[self.dataset_name] != 'None').tolist()]

    def next_data(self, data_type, index):
        """Final stage for preprosessing of data

           It determines wethere an array should be splitted and resized or simply resized
           before adding it to data_buffer.
        """
        if len(self.data_buffer[data_type]) == 0:
            if data_type is 'focal_length':
                data = self.load_data(
                    data_type, int(index / self.indexes['split_coefficient']))
                self.data_buffer[data_type] = [data]
            else:
                resize_info = 1
                if self.focal_length is not None:
                    array_focal_length = self.load_data(
                        'focal_length',
                        int(index / self.indexes['split_coefficient']))
                    resize_info = self.focal_length / array_focal_length
                elif self.split is False:
                    resize_info = self.output_shape

                image = self.load_data(
                    data_type, int(index / self.indexes['split_coefficient']),
                    resize_info)

                if self.split is True:
                    self.data_buffer[data_type] = resize_and_split(
                        image, self.output_shape, self.max_split)
                else:
                    self.data_buffer[data_type] = [image]

        return self.data_buffer[data_type].pop(0)

    def next(self):
        """Retrieve the next pairs from the dataset"""

        if self.indexes['current'] + self.batch_size > self.indexes['end']:
            if not self.repeater:
                raise StopIteration
            self.indexes['current'] = self.indexes['start']
        self.indexes['current'] = self.indexes['current'] + self.batch_size

        data_dict = dict()

        for data_type in self.data_list:
            self.data_history[data_type] = None
            data_dict[data_type] = np.array([
                self.next_data(data_type, i)
                for i in range(self.indexes['current'] -
                               self.batch_size, self.indexes['current'])
            ])

        feature_list = []
        for feature in self.feature_types:
            feature_list.append(data_dict[feature])
        label_list = []
        for label in self.label_types:
            if isinstance(data_dict[label], list):
                label_list.extend(data_dict[label])
            else:
                label_list.append(data_dict[label])

        return feature_list, label_list


class SynthiaSf(DatasetGenerator):
    """Iterator for looping over elements of SYNTHIA-SF backwards."""
    def __init__(self, dataset_dir, **kwargs):

        self.dataset_dir = dataset_dir
        self.available_categories = [
            'focal_length', 'image', 'segmentation', 'depth',
            'sparse_segmentation', 'semantic_depth'
        ]
        super().__init__(**kwargs)

    def data_frame_creator(self, usage, shuffle):
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

        focal_length = [
            847.630211643  # from dataset README.txt
            for _ in range(len(rgb_data))
        ]

        dataset = {
            'RGB': rgb_data,
            'DEPTH': depth_data,
            'SEGMENTATION': segmentation_data,
            'FOCAL_LENGTH': focal_length
        }

        if shuffle:
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
        features = np.array(features / 255., dtype=np.float32)

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
            labels = np.array(labels, dtype=np.float32)
        else:
            raise ValueError('invalid label type')

        return features, labels


class VIPER(DatasetGenerator):
    """Iterator for looping over elements of a VIPER(PlayForBenchmark) dataset ."""
    def __init__(self, dataset_dir, **kwargs):

        self.dataset_dir = dataset_dir
        super().__init__(**kwargs)

    def data_frame_creator(self, usage, shuffle):
        """Pandas dataFrame for addresses of images and corresponding labels"""

        if usage == 'train':
            main_folder = '/train'

        elif usage == 'validation':
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

        if shuffle:
            return pd.DataFrame(dataset).sample(
                frac=1, random_state=123).reset_index(drop=True)
        return pd.DataFrame(dataset)


class MAPILLARY(DatasetGenerator):
    """Iterator for looping over elements of a MAPILLARY dataset ."""
    def __init__(self, dataset_dir, **kwargs):

        self.dataset_dir = dataset_dir
        super().__init__(**kwargs)

    def data_frame_creator(self, usage, shuffle):
        """Pandas dataFrame for addresses of images and corresponding labels"""
        if usage == 'train':
            main_folder = '/training'

        elif usage == 'validation':
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

        if shuffle:
            return pd.DataFrame(dataset).sample(
                frac=1, random_state=123).reset_index(drop=True)
        return pd.DataFrame(dataset)


class CITYSCAPES(DatasetGenerator):
    """Iterator for looping over elements of a CITYSCAPES dataset ."""
    def __init__(self, dataset_dir, **kwargs):

        self.dataset_dir = dataset_dir
        super().__init__(**kwargs)

    def data_frame_creator(self, usage, shuffle):
        """Pandas dataFrame for addresses of images and corresponding labels"""

        if usage == 'train':
            main_folder = '/train'

        elif usage == 'validation':
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

        if shuffle:
            return pd.DataFrame(dataset).sample(
                frac=1, random_state=123).reset_index(drop=True)
        return pd.DataFrame(dataset)


class Lyft(DatasetGenerator):
    """Iterator for looping over elements of Lyft."""
    def __init__(self, dataset_dir, **kwargs):

        self.dataset_dir = dataset_dir
        self.available_categories = [
            'focal_length',
            'image',
            'depth',
        ]
        super().__init__(**kwargs)

    def data_frame_creator(self, usage, shuffle):
        """Loads Lyft's SDK"""
        level5data = LyftDataset(data_path=self.dataset_dir,
                                 json_path=self.dataset_dir + '/v1.02-train',
                                 verbose=False)
        setattr(level5data, 'shape', [int(len(level5data.sample) * 7)])
        return level5data

    def load_data(self, data_type, index, resize_info=1):
        sample_index = int(index / 7)
        # 'CAM_FRONT_ZOOMED' is removed because of different size and different focal length
        channel = ('CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT',
                   'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')[index % 7]
        explorer = LyftDatasetExplorer(self.dataset)
        lidar_token = self.dataset.sample[sample_index]['data']['LIDAR_TOP']
        cam_token = self.dataset.sample[sample_index]['data'][channel]
        sample_token = self.dataset.sample[sample_index]['token']
        points, coloring, image = explorer.map_pointcloud_to_image(
            lidar_token, cam_token)

        if data_type is "focal_length":
            for data in self.dataset.sample_data:
                if data['sample_token'] == sample_token and data[
                        'channel'] == channel:
                    calibrated_sensor_token = data['calibrated_sensor_token']

            for calibation_data in self.dataset.calibrated_sensor:
                if calibation_data['token'] == calibrated_sensor_token:
                    focal_length = calibation_data['camera_intrinsic'][0][0]
            return focal_length

        if data_type is "image":
            return resize(np.array(image), resize_info)

        if data_type is "depth":
            depth = np.zeros(np.array(image).shape[:2])  # pylint: disable=E1136
            for i, color in enumerate(coloring):
                depth[int(np.floor(points[1, :])[i]),
                      int(np.floor(points[0, :])[i])] = color

            if len(depth.shape) == 2:
                depth = np.expand_dims(depth, -1)

            return resize(depth, resize_info)
