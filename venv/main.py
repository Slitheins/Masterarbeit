import cv2
import numpy as np
import copy
import time
import os
import open3d as o3d
from operator import attrgetter
import open3d as o3d
import pickle
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'venv', 'src')))
from src.calibration import Camera, PhotoSet, CameraCalibration, StereoRectify
from src.stereorectify import RectifyImages
from src.disp_generation import stereo_matchSGBM
from src.pcd_generation import depth2xyz, visualization, load_params
from src.pcd_registration import draw_registration_result


if __name__ == '__main__':
    """Implementation of point cloud registration.
     
    4 steps will be executed.
    Step 1 : the parallel correction of image pairs obtained from a stereo camera pair using intrinsic/extrinsic matrices.
    Step 2 : disparity map generation using SGBM algorithm.
    Step 3 : point cloud generation using intrinsic and Q matrices.
    Step 4 : point cloud registration using homogenous matrix H.
    
    Notes
    -----
    Intrinsic, extrinsic, Q and H matrices can be loaded from previous saved pickle files.
    
    """

    print('Please choose the ID of the camera pair whose image pair need to be parallel rectified.')
    Nr1 = input("registrate_pcd_1:")
    Nr2 = input("registrate_pcd_2:")

    # get the current working directory.
    cur_dir = os.path.abspath(os.path.dirname((__file__)))
    print('cur_dir:', cur_dir)

    for i in (Nr1, Nr2):
        start = time.time()
        device_name = 'camera_' + i

        # get the directory where the images for further parallel rectification are stored.
        path = os.path.join(cur_dir, 'images', 'device' + str(i))
        print("Current operating path is:", path)
        #
        if os.path.exists(path):
            cm_path = os.path.join(path, 'intrinsic.pickle')
            q_path = os.path.join(path, 'extrinsic.pickle')
            rectify_model = pickle.load(open(q_path, 'rb'))
            # print("rectify:", rectify_model)
            # cam_matrix = pickle.load(open(cm_path, 'rb'))['M1']
            # Q = pickle.load(open(q_path, 'rb'))['Q']

        # read image pair used for parallel correction.
        for_rectify_l = cv2.imread(path + '\\' + 'left.jpg', -1)
        for_rectify_l = np.rot90(for_rectify_l, 3)
        for_rectify_r = cv2.imread(path + '\\' + 'right.jpg', -1)
        for_rectify_r = np.rot90(for_rectify_r, 3)

        # execute parallel correction.
        print('Starting parallel correction for camera pair %s' %(i))
        device_name = RectifyImages(rectify_model, path)
        device_name.rectify_image(for_rectify_l, for_rectify_r)
        result1 = cv2.imread(path + '\\' + 'result1.png', -1)
        result2 = cv2.imread(path + '\\' + 'result2.png', -1)
        # device_name.draw_line(result1, result2)

        # disparity map generation.
        print('Starting generating disparity map for camera pair %s' %(i))
        stereo_matchSGBM(path, result1, result2)

        # name the storage name of the point cloud.
        pcd_name = 'pcd' + i

        # convert the image to grayscale.
        result1 = cv2.cvtColor(result1, cv2.COLOR_BGR2RGB)
        disparity = cv2.imread(path + '\\' + 'disparity.png', -1)
        # verify that the files are loaded correctly.
        if disparity is None or result1 is None:
            print("Check path")
            exit()

        # load the camera matrix and Q matrix.
        depth_cam_matrix, Q = load_params(path)
        # convert disparity map to 3D points.
        points = depth2xyz(disparity, depth_cam_matrix, Q, with_Q=False)

        save_name = pcd_name + '.ply'
        print('save_name', save_name)

        if i == Nr1:
            source = visualization(points, path, save_name)
        else:
            target = visualization(points, path, save_name)
        # pointcloud = visualization(points, path, save_name)

    # performing a transformation on a point cloud will change the original point cloud.
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    print('still running')

    # get registration transformation.
    pl_reg_trans = os.path.join(cur_dir, 'src', 'registration_result' + '_' + str(Nr1) + '_' + str(Nr2) + '.pickle')
    print('loadname:', pl_reg_trans)
    icp_trans = pickle.load(open(pl_reg_trans, 'rb'))['ICP']
    draw_registration_result(source_temp, target_temp, icp_trans)
    # o3d.io.write_point_cloud("trans_of_source1.pcd", source_temp)#
    o3d.visualization.draw_geometries([source_temp, target_temp], width=600, height=600)

