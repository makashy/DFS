''' A module for visualizing data
'''
import os

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import plotly.express as px
import yaml
from IPython.display import display
from ipywidgets import interact, interactive_output

from utils import data_importer


def draw(array, title, class_name, class_index):
    ''' Draws different array types differently
    '''
    if title == 'focal_length':
        plt.text(
            0.5,
            0.5,
            "focal length(pixels):{}".format(array),
            fontsize=10,
            color='red',
            horizontalalignment='center',
            verticalalignment='center',
        )
        plt.axis('off')
        plt.title(title)
    if title == 'image':
        plt.imshow(array)
        plt.title(title)
    if title == 'sparse_segmentation':
        plt.imshow(array[:, :, 0])
        plt.title(title)
    if title == 'segmentation':
        plt.imshow(array[:, :, class_index])
        plt.title(title + ": " + class_name)
    if title == 'depth':
        plt.imshow(array[:, :, 0])
        plt.title(title)
    if title == 'semantic_depth':
        plt.imshow(array[:, :, class_index])
        plt.title(title + ": " + class_name)


def draw_samples(feature_list,
                 label_list,
                 predict_list,
                 feature_types: list,
                 label_types: list,
                 predict_types: list,
                 sample_num=None,
                 class_name='None',
                 class_index=None,
                 cmap="prism"):
    ''' Draws features and labels of a sample in separate rows
    '''
    n_columns = 3
    n_rows = max(len(feature_types), len(label_types))
    min_side = min(feature_list[0][0].shape[:2])
    plt.figure(figsize=(n_columns * 5 * feature_list[0][0].shape[1] / min_side,
                        n_rows * 5 * feature_list[0][0].shape[0] / min_side))
    plt.set_cmap(cmap)

    for i, title in enumerate(feature_types):
        plt.subplot(n_rows, n_columns, n_columns * i + 1)
        draw(feature_list[i][sample_num], title, class_name, class_index)

    for i, title in enumerate(label_types):
        plt.subplot(n_rows, n_columns, n_columns * i + 2)
        draw(label_list[i][sample_num], title, class_name, class_index)

    if len(predict_types) > 0:
        for i, title in enumerate(predict_types):
            plt.subplot(n_rows, n_columns, n_columns * i + 3)
            draw(predict_list[i][sample_num], title, class_name, class_index)


def generate_point_cloud(depth, image):
    """Generates a colored point cloud from an image and its corresponding depth map"""
    image_image = o3d.geometry.Image(image)
    depth_image = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        image_image, depth_image)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    rgb_vector = np.reshape(image, [-1, 3])
    pcd.colors = o3d.utility.Vector3dVector(rgb_vector)
    o3d.visualization.draw_geometries([pcd])


def generate_s_point_cloud(depth, segmentation):
    """Generates a semantically colored point cloud from a semantic segmentation and its
    corresponding depth map
    """
    segmentation_image = o3d.geometry.Image(segmentation)
    depth_image = o3d.geometry.Image(depth)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        segmentation_image, depth_image)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))

    rgb_vector = np.reshape(segmentation, [-1, 1])
    pcd.colors = o3d.utility.Vector3dVector(rgb_vector)
    o3d.visualization.draw_geometries([pcd])


def comp_dataframe(logs_root: str, metric_list: list):
    """Create a united dataframe that contains all logs available in logs_root.
    It also separates training logs from test logs to ease visualization of both
    them at the same time.

    Arguments:
        logs_root: a directory containing one or more folders of logs
        metric_list: a list of preferred metrics to be visualized
    """
    dirs = os.listdir(logs_root)
    log_list = []
    for folder_name in dirs:
        try:
            metric_log = pd.read_pickle(logs_root + '/' + folder_name +
                                        '/metric_log.pkl')
            model = folder_name

            for mode in ['test', 'training']:
                new_log = pd.DataFrame()
                new_log['epoch'] = metric_log.iloc[:,
                                                   metric_log.columns.
                                                   get_loc('epoch')]
                new_log['time'] = metric_log.iloc[:,
                                                  metric_log.columns.
                                                  get_loc('time')]
                new_log['Model'] = model
                new_log['Mode'] = mode
                for metric in metric_list:

                    if mode == 'test':
                        metric_name = 'val_' + metric
                    else:
                        metric_name = metric

                    if metric_name in metric_log.columns:
                        new_log[metric] = metric_log.iloc[:,
                                                          metric_log.columns.
                                                          get_loc(metric_name)]
                log_list.append(new_log)

        except FileNotFoundError:
            pass

    return pd.concat(log_list).reset_index(drop=True)


def smooth_logs(log: pd.DataFrame, alpha=0.5):
    """Applies a low pass filter to log data to smooth it.

    Arguments:
        log: log dataframe to bo filtered
        alpha: smoothing factor
    """
    # initialize dataframe
    log_filtered = pd.DataFrame(columns=log.columns)
    log_filtered.loc[0] = log.loc[0].values

    for i in range(1, len(log) - 1):
        log_filtered.loc[i] = log.loc[i].values
        if log_filtered.Model[i - 1] == log_filtered.Model[i]:
            log_filtered.iloc[i, 4:] = log_filtered.iloc[i - 1, 4:] * (
                1 - alpha) + log_filtered.iloc[i, 4:] * alpha
    return log_filtered


class MetricsDashboard():
    """ Dashboard for visualizing metrics logs"""
    def __init__(self, log_dir, metric_list):
        self.log_dataframe = comp_dataframe(log_dir, metric_list)
        self.metric_list = metric_list

    def plot_metrics(self, y, log_y, alpha):
        """Plots metrics logs by plotly """
        smoothed_log = smooth_logs(self.log_dataframe, alpha)
        fig = px.line(smoothed_log,
                      x='epoch',
                      y=y,
                      line_dash='Mode',
                      color='Model',
                      log_y=log_y)
        fig.write_html("report" + ".html")
        fig.show()

    def __call__(self):

        metric_type = widgets.Dropdown(options=self.metric_list,
                                       value=self.metric_list[0],
                                       description='Metric Type:',
                                       disabled=False)

        log_y = widgets.Checkbox(value=True, description='log_y')

        style = {'description_width': 'initial'}
        smoothing_factor = widgets.FloatSlider(min=0,
                                               max=1,
                                               value=1,
                                               step=0.05,
                                               description='smoothing factor',
                                               style=style)

        widget_group = widgets.HBox([metric_type, log_y, smoothing_factor])

        output = interactive_output(self.plot_metrics, {
            'y': metric_type,
            'log_y': log_y,
            'alpha': smoothing_factor
        })

        display(widget_group, output)


class SamplesDashboard():
    """ Dashboard for visualizing predicted samples in logs
    """
    def __init__(self, log_dir, label_table_address):
        self.log_dir = log_dir
        self.log_folders = self.check_folders(log_dir)
        self.label_table = data_importer.load_label_table(label_table_address)

        self.configurations = None
        self.feature_list = None
        self.label_list = None
        self.predicted_samples = None
        self.class_ids = None

    def check_folders(self, log_dir):
        """ Searches log_dir for folders that contain log files
        and returns a list of those folders.
        """
        dirs = os.listdir(log_dir)
        log_folders = []
        for folder in dirs:
            if 'configurations.yaml' in os.listdir(
                    os.path.join(log_dir, folder)):
                log_folders.append(folder)
        return log_folders

    def draw_selected(self, num_slider, layer_slider, epoch_slider):
        """Draws selected samples
        """
        num_slider -= 1
        layer_slider -= 1
        epoch_slider -= 1
        class_name = self.label_table.loc[self.class_ids[layer_slider]][0]
        draw_samples(
            self.feature_list[0],
            self.label_list[0],
            self.predicted_samples[epoch_slider],
            feature_types=self.configurations['dataset']['feature_types'],
            label_types=self.configurations['dataset']['label_types'],
            predict_types=self.configurations['dataset']['label_types'],
            sample_num=num_slider,
            class_name=class_name,
            class_index=layer_slider)

    def load_selected_log(self, folder):
        """Loads selected log files and and displays widgets for selecting
        samples.
        """
        complete_dir = os.path.join(self.log_dir, folder, 'visual_log.pkl')

        visual_log = pd.read_pickle(complete_dir)

        stream = open(
            os.path.join(self.log_dir, folder, 'configurations.yaml'), 'r')
        self.configurations = yaml.load(stream, Loader=yaml.FullLoader)
        stream.close()

        self.feature_list = visual_log['feature_list']
        self.label_list = visual_log['label_list']
        self.predicted_samples = [
            visual_log[index] for index in visual_log.columns[2:]
        ]
        self.class_ids = data_importer.available_classes(
            self.label_table, self.configurations['dataset']
            ['dataset_name']) if self.configurations['dataset'][
                'class_ids'] == 'all' else self.configurations['dataset'][
                    'class_ids']

        num_slider = widgets.IntSlider(
            description="Sample Num",
            min=1,
            max=self.configurations['dataset']['batch_size'])
        num_play = widgets.Play(
            interval=1000,
            min=1,
            max=self.configurations['dataset']['batch_size'],
            step=1,
            description="Sample Num")
        widgets.jslink((num_slider, 'value'), (num_play, 'value'))
        num_box = widgets.VBox([num_slider, num_play])

        layer_slider = widgets.IntSlider(description="Sample layer",
                                         min=1,
                                         max=len(self.class_ids))
        layer_play = widgets.Play(interval=1000,
                                  min=1,
                                  max=len(self.class_ids),
                                  step=1,
                                  description="Sample layer")
        widgets.jslink((layer_slider, 'value'), (layer_play, 'value'))
        layer_box = widgets.VBox([layer_slider, layer_play])

        epoch_slider = widgets.IntSlider(description="Epoch Num",
                                         min=1,
                                         max=len(visual_log.columns) - 2)
        epoch_play = widgets.Play(interval=500,
                                  min=1,
                                  max=len(visual_log.columns),
                                  step=1,
                                  description="Epoch Num")
        widgets.jslink((epoch_slider, 'value'), (epoch_play, 'value'))
        epoch_box = widgets.VBox([epoch_slider, epoch_play])

        settings_box = widgets.HBox([num_box, layer_box, epoch_box])

        output = interactive_output(
            self.draw_selected, {
                'num_slider': num_slider,
                'layer_slider': layer_slider,
                'epoch_slider': epoch_slider
            })

        display(settings_box, output)

    def __call__(self):
        """Displays widgets for selecting a log folder.
        This method should be called to display samples.
        """
        log_selection = widgets.Dropdown(options=self.log_folders,
                                         description='Select log:')

        interact(self.load_selected_log, folder=log_selection)
