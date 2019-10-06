''' A module for visualizing data
'''
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d


def one_hot2sparse(one_hot):
    ''' Convers an one_hot label to a sparse label
    '''
    sparse = np.ndarray(shape=one_hot.shape[0:2])
    for i in range(one_hot.shape[0]):
        for j in range(one_hot.shape[1]):
            sparse[i, j] = one_hot[i, j].argmax()
    return sparse


def draw(array, array_type):
    ''' Draws different array types differently
    '''
    if array_type == 'image':
        plt.imshow(array)
    if array_type == 'sparse_segmentation':
        plt.imshow(array)
    if array_type == 'segmentation':
        plt.imshow(one_hot2sparse(array))
        plt.set_cmap('prism')
    if array_type == 'depth':
        plt.imshow(array[:, :, 0])


def draw_samples(feature_list, label_list, index, feature_types, label_types):
    ''' Draws features and labels of a sample in separate rows
    '''
    n_columns = max(len(feature_types), len(label_types))
    plt.figure(figsize=(n_columns * 5, 8))
    for i, title in enumerate(feature_types):
        plt.subplot(2, n_columns, i + 1)
        draw(feature_list[i][index], title)
        plt.title(title)

    for i, title in enumerate(label_types):
        plt.subplot(2, n_columns, n_columns + i + 1)
        draw(label_list[i][index], title)
        plt.title(title)


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
