import cv2
import numpy as np
import glob
import time
import os
from operator import attrgetter
import matplotlib.pyplot as plt
from PIL import Image
import open3d as o3d
from src.calibration import Camera, PhotoSet, CameraCalibration
from src.stereorectify import StereoRectify
from src.depthmap import stereo_matchSGBM, stereo_matchBM



if __name__ == '__main__':

    print('Please choose the camera pair, which needed to be calibrated.')
    a = input("device:")
    device_name = 'camera_' + a
    # print('Calibrating:', device_name)
    device_name = Camera(a)
    path, path_l, path_r = device_name.get_dir
    # print('path', path)
    read_all = PhotoSet(path, path_l, path_r)
    imma1, imma2 = read_all.image_to_matrix
    square = CameraCalibration("square", imma1, imma2, path)
    camera_model = square.method_manager(9, 6, 2.5000000372529030e-02)
    print('camera1:', camera_model)

    # camera_model = {'rms': 0.6160846063201201,
    #  'M1': np.array([[8.63936278e+03, 0.00000000e+00, 2.14338706e+03],
    #                  [0.00000000e+00, 8.66314910e+03, 1.37603525e+03],
    #                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    #  'M2': np.array([[8.55432393e+03, 0.00000000e+00, 2.22009028e+03],
    #                  [0.00000000e+00, 8.57157453e+03, 1.62701783e+03],
    #                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]),
    #  'dist1': np.array([[1.08458092e-03, -1.46765523e+00, -4.31746568e-03, 8.08144707e-04, 1.26288403e+01]]),
    #  'dist2': np.array([[-4.01207645e-02, -4.86349653e-01, 3.39117551e-03, -9.56593330e-05, 4.24637107e+00]]),
    #  'R': np.array([[0.99943117, -0.00303017, -0.03358791],
    #                 [-0.00371039, 0.98002908, -0.19881962],
    #                 [0.03351959, 0.19883115, 0.97946037]]),
    #  'T': np.array([[-0.11844889], [1.6333303], [0.04970491]]),
    #  'E': np.array([[0.05493298, 0.27604469, 1.6096646], [0.05364699, 0.02340072, 0.11434651],
    #                 [-1.63196172, -0.11113409, 0.07841012]]),
    #  'F': np.array([[9.33392036e-09, 4.67752622e-08, 2.27854392e-03],
    #                 [9.09706623e-09, 3.95722786e-09, 1.42573854e-04],
    #                 [-2.40758814e-03, -2.71374248e-04, 1.00000000e+00]]),
    #  'dims': (4096, 3072)}

    stereo_device = 'stereo' + a
    stereo_device = StereoRectify(camera_model, path)
    rectify_model = stereo_device.get_rectify_transform
    for_rectify_l = cv2.imread('C:\\Users\\wyfmi\\Downloads\\opencv-4.5.5\\opencv-4.5.5\\samples\\data\\aloeL.jpg', 0)
    for_rectify_r = cv2.imread('C:\\Users\\wyfmi\\Downloads\\opencv-4.5.5\\opencv-4.5.5\\samples\\data\\aloeR.jpg', 0)
    stereo_device.rectify_image(for_rectify_l, for_rectify_r)
    stereo_device.draw_line(cv2.imread(path + '\\' + 'result1.png'), cv2.imread(path + '\\' + 'result2.png'))

    stereo_matchBM(cv2.imread(path + '\\' + 'result1.png'), cv2.imread(path + '\\' + 'result2.png'), path)