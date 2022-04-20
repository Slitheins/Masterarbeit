# !/usr/bin/env python
# -*- coding:utf-8 -*- 

"""
author: yifeng
date  : 27.03.2022
"""
import cv2
import numpy as np
import open3d as o3d


def depth2xyz(depth_map, depth_cam_matrix, flatten=False, depth_scale=1000):
    print('h, w:', depth_map.shape)
    fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
    cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
    h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
    z = depth_map / depth_scale
    # print('z:', z)
    x = (w - cx) * z / fx
    y = (h - cy) * z / fy
    xyz = np.dstack((x, y, z)) if flatten == False else np.dstack((x, y, z)).reshape(-1, 3)
    # xyz=cv2.rgbd.depthTo3d(depth_map,depth_cam_matrix)
    return xyz

img = cv2.imread('./result1.png', -1)
# ret, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
# img = cv2.imread('C:\\Users\\wyfmi\\Desktop\\2022-02-25_Kallibrierbilder\\result1')
depth = cv2.imread('./depthmap.png', -1)
# depth = cv2.imread('./Disparity.png', -1)

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
colors = img.reshape(-1, 3)

# depth_cam_matrix = np.array([[8.63936278e+03, 0.00000000e+00, 2.14338706e+03],
#                           [0.00000000e+00, 8.66314910e+03, 1.37603525e+03],
#                           [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
depth_cam_matrix = np.array([[8.03114872e+03, 0.00000000e+00, 2.28022179e+03],
                       [0.00000000e+00, 8.03197699e+03, 1.40290208e+03],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# Q = np.array([[1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.16516658e+03],
#               [0.00000000e+00,  1.00000000e+00,  0.00000000e+00, -2.85822198e+03],
#               [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 8.26171663e+03],
#               [0.00000000e+00,  0.00000000e+00, -6.10361337e-01, 9.47790013e+02]])

points = depth2xyz(depth, depth_cam_matrix, True)

index = np.where(points[:, 0] > 0.15)
points = np.delete(points, index, axis=0)
colors = np.delete(colors, index, axis=0)

index = np.where(points[:, 0] < -0.15)
points = np.delete(points, index, axis=0)
colors = np.delete(colors, index, axis=0)

index = np.where(points[:, 1] > 0.1)
points = np.delete(points, index, axis=0)
colors = np.delete(colors, index, axis=0)

index = np.where(points[:, 1] < -0.2)
points = np.delete(points, index, axis=0)
colors = np.delete(colors, index, axis=0)
#
index = np.where(points[:, 2] > 0.7)
points = np.delete(points, index, axis=0)
colors = np.delete(colors, index, axis=0)

index = np.where(points[:, 2] < 0.1)
points = np.delete(points, index, axis=0)
colors = np.delete(colors, index, axis=0)

point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)
point_cloud.colors = o3d.utility.Vector3dVector(colors / 255)

# point_cloud 需要用中括号括住

# downpcd = points.voxel_down_sample(voxel_size=0.05)
# o3d.visualization.draw_geometries([downpcd])
o3d.visualization.draw_geometries([point_cloud])
