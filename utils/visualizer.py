''' A module for visualizing data
'''
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


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
                 sample_num=None,
                 feature_types=[],
                 label_types=[],
                 predict_types=[],
                 class_name='None',
                 class_index=None,
                 cmap="prism"):
    ''' Draws features and labels of a sample in separate rows
    '''
    n_rows = 3
    n_columns = max(len(feature_types), len(label_types))
    min_side = min(feature_list[0][0].shape[:2])
    plt.figure(figsize=(n_columns * 5 * feature_list[0][0].shape[1] / min_side,
                        n_rows * 5 * feature_list[0][0].shape[0] / min_side))
    plt.set_cmap(cmap)

    for i, title in enumerate(feature_types):
        plt.subplot(n_rows, n_columns, i + 1)
        draw(feature_list[i][sample_num], title, class_name, class_index)

    for i, title in enumerate(label_types):
        plt.subplot(n_rows, n_columns, n_columns + i + 1)
        draw(label_list[i][sample_num], title, class_name, class_index)

    if len(predict_types) > 0:
        for i, title in enumerate(predict_types):
            plt.subplot(n_rows, n_columns, 2 * n_columns + i + 1)
            draw(predict_list[i][sample_num], title, class_name, class_index)


def draw_point_cloud(depth, image):
    a = o3d.geometry.Image(image)  #pylint: disable=no-member
    b = o3d.geometry.Image(depth)  #pylint: disable=no-member
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(a, b)  #pylint: disable=no-member
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(  #pylint: disable=no-member
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(  #pylint: disable=no-member
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))  #pylint: disable=no-member

    rgb_vector = np.reshape(image, [-1, 3])
    pcd.colors = o3d.utility.Vector3dVector(rgb_vector)  #pylint: disable=no-member
    o3d.visualization.draw_geometries([pcd])  #pylint: disable=no-member


def draw_s_point_cloud(depth, segmentation):
    a = o3d.geometry.Image(segmentation)  #pylint: disable=no-member
    b = o3d.geometry.Image(depth)  #pylint: disable=no-member
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(a, b)  #pylint: disable=no-member
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(  #pylint: disable=no-member
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(  #pylint: disable=no-member
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))  #pylint: disable=no-member

    rgb_vector = np.reshape(segmentation, [-1, 1])
    pcd.colors = o3d.utility.Vector3dVector(rgb_vector)  #pylint: disable=no-member
    o3d.visualization.draw_geometries([pcd])  #pylint: disable=no-member
