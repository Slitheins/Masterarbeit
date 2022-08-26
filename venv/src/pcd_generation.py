# !/usr/bin/env python
# -*- coding:utf-8 -*-

"""
author: yifeng
date  : 27.03.2022
"""
import cv2
import numpy as np
import open3d as o3d
import time
import os
import pickle


def depth2xyz(disparity, cam_matrix, Q, depth_scale=1000, flatten=False, with_Q=True):
    """Convert the 2D image to 3D point cloud.

    convert disparity map to 3D point cloud using camera intrinsic parameters and Q matrix..

    Parameters
    ----------
    disparity : array
        disparity map. The better the disparity map, the more accurate the point cloud is.
    cam_matrix : 4 x 4 matrix
        intrinsic parameters of the left camera.
    Q : 4×4 matrix
        disparity-to-depth mapping matrix.
    flatten : bool
        stack all the points in point cloud together.
    with_Q : bool
        if the OpenCV Function reprojectImageTo3D is used, Q matrix must be given.

    Returns
    -------
    xyz : ply
        point cloud with ply format.
        the points number are rows * columns of the image size.

    """
    if with_Q == True:
        fx, fy = cam_matrix[0, 0], cam_matrix[1, 1]
        cx, cy = cam_matrix[0, 2], cam_matrix[1, 2]
        h, w = np.mgrid[0:disparity.shape[0], 0:disparity.shape[1]]
        z = disparity / depth_scale
        x = (w - cx) * z / fx
        y = (h - cy) * z / fy
        xyz = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)
    else:
        threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
        xyz = np.reshape(threeD, (-1, 3))

    return xyz


def visualization(points, path, save_name):
    """Visualize point cloud. Initial filtering of point cloud.

    Preprocessing of the point cloud, deleting the outliers with z-axis value = 1000.
    Save the point cloud as ".ply" file.

    Parameters
    ----------
    points: ply
        point cloud.

    Returns
    -------
    pcd: ply
        the initial filtered point cloud.

    """

    # The initial filtering is carried out based on the following criteria.
    # Distance from the camera to objects are normally larger than 6 meters.
    # Noise equal to 1000 are defined in OpenCV.
    index = np.where(points[:, 2] < 6)
    # index = list(index)
    points = np.delete(points, index, axis=0)

    # z=1000 is definiert by OpenCV as outlier points.
    index = np.where(points[:, 2] > 999)
    points = np.delete(points, index, axis=0)

    # # visualization using open3D.
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # o3d.visualization.draw_geometries([pcd])

    # save point cloud
    o3d.io.write_point_cloud(path + '\\' + save_name, pcd)
    print('%s is saving' %(save_name))

    return pcd

def load_params(path):
    """Load camera matrix and Q matrix used for point cloud converting.

    Load the camera matrix and Q matrix for point cloud generation.

    Parameters
    ----------
    path : string
        path to save point clouds.

    """
    if os.path.exists(path):
        cm_path = os.path.join(path, 'intrinsic.pickle')
        q_path = os.path.join(path, 'extrinsic.pickle')
        depth_cam_matrix = pickle.load(open(cm_path, 'rb'))['M1']
        Q = pickle.load(open(q_path, 'rb'))['Q']
    else:
        print('Check the path')
    print('camera matrix and Q', depth_cam_matrix, Q)
    return depth_cam_matrix, Q


####################################################################################################################
# # The following sentences are used for point cloud generation.
# # If you want to test the result or do some preprocessing filtering, please activate the following sentence.

# start = time.time()
#
# # img = cv2.imread('./result1.png', -1)
#
# # depth = cv2.imread('./depthmap.png', -1)
#
# print('Please enter the ID of the camera pair that needs to be converted to point cloud.')
# print('Please make sure that the parallel corrected images and the disparity map have been generated '
#       'and saved in the same folder.')
# a = input("Pointcloud:")
# pcd_name = 'pointcloud' + a
# print("plname:", pcd_name)
# # get the current working directory.
# cur_dir = os.path.abspath(os.path.dirname((__file__)))
# print('cur_dir:', cur_dir)
# # get upper directory.
# upper = os.path.abspath(os.path.join(os.getcwd(), ".."))
# print('cur_dir2:', upper)
# img_path = os.path.join(upper, 'images', 'device' + str(a), 'result1.png')
# disparity_path = os.path.join(upper, 'images', 'device' + str(a), 'depthmap.png')
# # Read rectified left image and disparity map to generate the point cloud.
# img = cv2.imread(img_path, -1)
# depth = cv2.imread(disparity_path, -1)
# if img is None or depth is None:
#     print("Check path")
#     exit()
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
# # colors = img.reshape(-1, 3)
#
# params_path = os.path.join(upper, 'images', 'device' + str(a))
#
# depth_cam_matrix, Q = load_params(params_path)
#
#
# points = depth2xyz(depth, depth_cam_matrix, Q, with_Q=False)
#
# visualization(points, params_path)
#
#
# print("Point cloud transformation took %.3f sec.\n" % (time.time() - start))
